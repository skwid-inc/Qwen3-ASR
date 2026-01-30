# Qwen3-ASR

Multilingual speech recognition with streaming support, deployed on Modal with vLLM backend.

## Models

| Model | Size | Languages |
|-------|------|-----------|
| Qwen3-ASR-1.7B | 1.7B | 52 (30 languages + 22 Chinese dialects) |
| Qwen3-ASR-0.6B | 0.6B | 52 (30 languages + 22 Chinese dialects) |

Supported languages include: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Hungarian, Macedonian, Romanian.

## Modal Deployment

Both models are deployed on Modal with WebSocket streaming endpoints.

### Endpoints

| Model | WebSocket URL |
|-------|---------------|
| Qwen3-ASR-1.7B | `wss://salient--qwen3-asr-1-7b-streaming-qwen3asr-webapp.modal.run/ws` |
| Qwen3-ASR-0.6B | `wss://salient--qwen3-asr-0-6b-streaming-qwen3asr-webapp.modal.run/ws` |

### Deploy

```bash
# Deploy 1.7B model
modal deploy modal/qwen3_asr_modal.py

# Deploy 0.6B model
modal deploy modal/qwen3_asr_0_6b_modal.py
```

### WebSocket Protocol

1. Connect to the WebSocket endpoint
2. Wait for `READY` message
3. Send audio chunks as raw bytes (16kHz, 16-bit signed int16 PCM)
4. Receive JSON responses: `{"language": "English", "text": "...", "is_final": false}`
5. Send `END` text message when done
6. Receive final response with `is_final: true`

### Test Client

```bash
python modal/test_streaming.py wss://YOUR_MODAL_URL/ws /path/to/audio.wav
```

Example output:
```
Connected to server
Server: READY
Sent 1600 samples
Partial: {"language": "English", "text": "Hello", "is_final": false}
...
Final: {"language": "English", "text": "Hello world this is a test", "is_final": true}
```

## Local Development

### Installation

```bash
pip install -e ".[vllm]"
```

### Quick Test

```python
from qwen_asr import Qwen3ASRModel

if __name__ == '__main__':
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.85,
    )

    results = model.transcribe(audio=["test.wav"])
    print(results[0].language, results[0].text)
```

## Fine-tuning

See [finetuning/](finetuning/) for instructions. Dataset format is JSONL:

```json
{"audio": "/path/to/audio.wav", "text": "transcription", "prompt": "optional system prompt"}
```

Run with:
```bash
python finetuning/qwen3_asr_sft.py --train_file train.jsonl --model_path Qwen/Qwen3-ASR-1.7B
```

## Original Repository

This is a fork focused on Modal deployment. For full documentation, evaluation benchmarks, and additional features (forced aligner, Gradio demo, Docker), see the original: [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
