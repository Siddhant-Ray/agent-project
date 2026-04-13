from __future__ import annotations

DTYPE_BYTES = {
    'float32': 4,
    'float16': 2,
    'bfloat16': 2,
}

# Metadata validated from each model's published Hugging Face config.
MODEL_CATALOG = {
    'nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct': {
        'family': 'llama',
        'display_name': 'nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct',
        'max_context_tokens': 1_073_152,
        'num_hidden_layers': 32,
        'num_key_value_heads': 8,
        'head_dim': 128,
        'torch_dtype': 'float32',
    },
    'Qwen/Qwen2.5-7B-Instruct-1M': {
        'family': 'qwen2',
        'display_name': 'Qwen/Qwen2.5-7B-Instruct-1M',
        'max_context_tokens': 1_010_000,
        'num_hidden_layers': 28,
        'num_key_value_heads': 4,
        'head_dim': 128,  # hidden_size 3584 / num_attention_heads 28
        'torch_dtype': 'bfloat16',
    },
    'Qwen/Qwen2.5-14B-Instruct-1M': {
        'family': 'qwen2',
        'display_name': 'Qwen/Qwen2.5-14B-Instruct-1M',
        'max_context_tokens': 1_010_000,
        'num_hidden_layers': 48,
        'num_key_value_heads': 8,
        'head_dim': 128,  # hidden_size 5120 / num_attention_heads 40
        'torch_dtype': 'bfloat16',
    },
}

DEFAULT_MODEL = 'nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct'


def get_model_info(model_name: str) -> dict:
    if model_name not in MODEL_CATALOG:
        raise ValueError(f'Unsupported model: {model_name}')
    info = dict(MODEL_CATALOG[model_name])
    bytes_per_element = DTYPE_BYTES[info['torch_dtype']]
    info['bytes_per_element'] = bytes_per_element
    info['bytes_per_token'] = (
        info['num_hidden_layers']
        * info['num_key_value_heads']
        * info['head_dim']
        * 2
        * bytes_per_element
    )
    return info


def model_options() -> list[dict]:
    return [
        {
            'name': name,
            'display_name': data['display_name'],
            'max_context_tokens': data['max_context_tokens'],
        }
        for name, data in MODEL_CATALOG.items()
    ]
