from __future__ import annotations

from model_catalog import DEFAULT_MODEL, get_model_info


def kv_bytes_from_tokens(tokens: float, model_name: str = DEFAULT_MODEL) -> float:
    info = get_model_info(model_name)
    return tokens * info['bytes_per_token']


def format_binary_bytes(num_bytes: float) -> str:
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    value = float(num_bytes)
    unit_idx = 0
    while value >= 1024 and unit_idx < len(units) - 1:
        value /= 1024
        unit_idx += 1
    if value >= 100:
        return f'{value:.0f} {units[unit_idx]}'
    if value >= 10:
        return f'{value:.1f} {units[unit_idx]}'
    return f'{value:.2f} {units[unit_idx]}'
