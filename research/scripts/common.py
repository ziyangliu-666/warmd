"""Shared model definition for the warmd investigation: real Llama graphs with random weights."""
import json, os, time
LAYERS = {2: 2, 8: 13, 16: 28}

def config(gb):
    from transformers import LlamaConfig
    return LlamaConfig(hidden_size=4096, intermediate_size=16384, num_hidden_layers=LAYERS[gb],
                       num_attention_heads=32, vocab_size=32000, torch_dtype="float16",
                       attn_implementation="sdpa")

def path(gb):
    return f"/workspace/m/{gb}/model.safetensors"

def emit(**kw):
    print(json.dumps(kw), flush=True)
