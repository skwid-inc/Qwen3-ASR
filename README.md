# Qwen3-ASR Modal Deployment

Streaming ASR on Modal with WebSocket endpoints.

## Deploy

```bash
# 1.7B model
modal deploy modal/qwen3_asr_modal.py

# 0.6B model
modal deploy modal/qwen3_asr_0_6b_modal.py
```

## WebSocket Protocol

1. Connect to `wss://<modal-url>/ws`
2. Wait for `READY`
3. Send audio chunks (16kHz, 16-bit PCM)
4. Receive JSON: `{"language": "...", "text": "...", "is_final": false}`
5. Send `END` when done
6. Receive final response with `is_final: true`

## Test

```bash
python modal/test_streaming.py wss://YOUR_URL/ws /path/to/audio.wav
```
