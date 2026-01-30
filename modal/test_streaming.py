import asyncio
import websockets
import numpy as np
import soundfile as sf


async def test_streaming(ws_url: str, audio_path: str):
    # Load and convert audio to 16kHz int16
    wav, sr = sf.read(audio_path, dtype="float32")
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    pcm_int16 = (wav * 32767).astype(np.int16)

    async with websockets.connect(ws_url) as ws:
        ready = await ws.recv()
        print(f"Server: {ready}")

        # Stream in 100ms chunks (1600 samples @ 16kHz)
        chunk_size = 1600
        for i in range(0, len(pcm_int16), chunk_size):
            chunk = pcm_int16[i : i + chunk_size].tobytes()
            await ws.send(chunk)

            result = await ws.recv()
            print(f"Partial: {result}")

            await asyncio.sleep(0.05)  # Simulate real-time

        await ws.send("END")
        try:
            final = await ws.recv()
            print(f"Final: {final}")
        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed by server (expected)")


if __name__ == "__main__":
    import sys

    ws_url = sys.argv[1]  # e.g., wss://your-app--qwen3-asr-streaming.modal.run/ws
    audio_path = sys.argv[2]
    asyncio.run(test_streaming(ws_url, audio_path))
