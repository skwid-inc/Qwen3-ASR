import modal

APP_NAME = "qwen3-asr-0-6b-streaming"
CACHE_PATH = "/root/.cache/huggingface"
MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE_PATH})
    .apt_install("git", "libsndfile1", "ffmpeg", "sox")
    .pip_install(
        "hf_transfer",
        "vllm==0.14.0",
        "transformers==4.57.6",
        "accelerate==1.12.0",
        "qwen-omni-utils",
        "soundfile",
        "librosa",
        "fastapi[standard]",
        # Dependencies for forced aligner (required by import chain)
        "nagisa==0.2.11",
        "soynlp==0.0.493",
    )
    .add_local_dir("qwen_asr", "/root/qwen_asr", copy=True)  # Copy local package
)


@app.cls(
    volumes={CACHE_PATH: model_cache},
    gpu="H100",
    image=image,
    timeout=3600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=64)
class Qwen3ASR:
    @modal.enter()
    def load(self):
        import sys
        import numpy as np

        sys.path.insert(0, "/root")
        from qwen_asr import Qwen3ASRModel

        self.asr = Qwen3ASRModel.LLM(
            model=MODEL_NAME,
            gpu_memory_utilization=0.85,
            max_new_tokens=64,  # Lower for streaming latency
        )
        self._warmup()

    def _warmup(self):
        """GPU warmup with dummy audio"""
        import numpy as np

        test_audio = np.zeros(16000 * 3, dtype=np.float32)  # 3s silence
        for _ in range(3):
            state = self.asr.init_streaming_state(chunk_size_sec=2.0)
            self.asr.streaming_transcribe(test_audio[:32000], state)
            self.asr.finish_streaming_transcribe(state)
        print("Warmup complete")

    @modal.asgi_app()
    def webapp(self):
        import numpy as np
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        web_app = FastAPI()

        asr = self.asr  # Capture reference for closure

        @web_app.websocket("/ws")
        async def websocket_transcribe(ws: WebSocket):
            await ws.accept()
            await ws.send_text("READY")

            state = asr.init_streaming_state(
                unfixed_chunk_num=2,
                unfixed_token_num=5,
                chunk_size_sec=2.0,
            )

            try:
                while True:
                    msg = await ws.receive()

                    if "bytes" in msg:
                        # Convert int16 PCM to float32
                        pcm_int16 = np.frombuffer(msg["bytes"], dtype=np.int16)
                        pcm_float = pcm_int16.astype(np.float32) / 32768.0

                        asr.streaming_transcribe(pcm_float, state)

                        await ws.send_json({
                            "language": state.language,
                            "text": state.text,
                            "is_final": False,
                        })

                    elif "text" in msg and msg["text"] == "END":
                        asr.finish_streaming_transcribe(state)
                        await ws.send_json({
                            "language": state.language,
                            "text": state.text,
                            "is_final": True,
                        })
                        break

            except WebSocketDisconnect:
                pass
            finally:
                await ws.close()

        @web_app.get("/health")
        def health():
            return {"status": "ok"}

        return web_app

    @modal.method()
    def transcribe_batch(self, audio_list: list[bytes]) -> list[dict]:
        """Batch transcription for non-streaming use"""
        import numpy as np

        results = []
        for audio_bytes in audio_list:
            pcm = np.frombuffer(audio_bytes, dtype=np.int16)
            pcm_float = pcm.astype(np.float32) / 32768.0

            out = self.asr.transcribe(audio=[(pcm_float, 16000)])
            results.append({
                "language": out[0].language,
                "text": out[0].text,
            })
        return results
