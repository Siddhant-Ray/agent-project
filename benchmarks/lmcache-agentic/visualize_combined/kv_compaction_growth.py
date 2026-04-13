from __future__ import annotations

from typing import Dict

from kv_memory import format_binary_bytes, kv_bytes_from_tokens
from model_catalog import get_model_info
from prefix_cache_cost import load_session_rows


def compute_kv_compaction_curve(csv_path: str, target_session_id: str, compaction_fraction: float, model_name: str) -> Dict:
    if not (0.0 <= compaction_fraction <= 1.0):
        raise ValueError('compaction_fraction must be between 0.0 and 1.0')

    rows = load_session_rows(csv_path, target_session_id)
    if not rows:
        raise ValueError(f"No rows found for session_id={target_session_id!r}")

    info = get_model_info(model_name)
    first_prefix = rows[0]['prefix_length']
    first_kv_bytes = kv_bytes_from_tokens(first_prefix, model_name)
    points = [
        {
            'turn': 0,
            'label': 'round0',
            'prefix_tokens': first_prefix,
            'compacted_prefix_tokens': first_prefix,
            'new_tokens': 0,
            'kv_tokens': first_prefix,
            'kv_bytes': first_kv_bytes,
            'kv_display': format_binary_bytes(first_kv_bytes),
        }
    ]

    for idx, row in enumerate(rows, start=1):
        prefix_len = row['prefix_length']
        extra_len = row['next_turn_added_length']
        compacted_prefix_len = prefix_len * compaction_fraction
        kv_tokens = compacted_prefix_len + extra_len
        kv_bytes = kv_bytes_from_tokens(kv_tokens, model_name)
        points.append(
            {
                'turn': idx,
                'label': row['turn_indices'],
                'prefix_tokens': prefix_len,
                'compacted_prefix_tokens': compacted_prefix_len,
                'new_tokens': extra_len,
                'kv_tokens': kv_tokens,
                'kv_bytes': kv_bytes,
                'kv_display': format_binary_bytes(kv_bytes),
                'compaction_fraction': compaction_fraction,
            }
        )

    max_kv_bytes = max(p['kv_bytes'] for p in points)
    return {
        'session_id': target_session_id,
        'mode': 'kv_compaction',
        'model': model_name,
        'kv_dtype': info['torch_dtype'],
        'bytes_per_token': info['bytes_per_token'],
        'compaction_fraction': compaction_fraction,
        'points': points,
        'max_kv_tokens': max(p['kv_tokens'] for p in points),
        'max_kv_bytes': max_kv_bytes,
        'max_kv_display': format_binary_bytes(max_kv_bytes),
    }
