from __future__ import annotations

from prefix_cache_cost import load_session_rows
from model_catalog import get_model_info
from gpu_catalog import get_gpu_info

GAP_SECONDS = 10.0


def compute_cost_from_tokens(tokens: float, throughput_tokens_per_second: float, cost_per_second_usd: float) -> float:
    return (tokens / throughput_tokens_per_second) * cost_per_second_usd


def model_weight_bytes_estimate(model_name: str) -> int:
    if '14B' in model_name:
        params = 14_000_000_000
    elif '8B' in model_name:
        params = 8_000_000_000
    elif '7B' in model_name:
        params = 7_000_000_000
    else:
        params = 8_000_000_000
    info = get_model_info(model_name)
    return params * info['bytes_per_element']


def usable_kv_memory_bytes(model_name: str, gpu_name: str) -> float:
    gpu = get_gpu_info(gpu_name)
    gross = gpu['total_memory_bytes'] * gpu['gpu_memory_utilization']
    kv_bytes = gross - model_weight_bytes_estimate(model_name) - gpu['reserved_runtime_bytes']
    if kv_bytes <= 0:
        raise ValueError('Usable KV memory is <= 0 for this model/GPU preset')
    return float(kv_bytes)


def storage_cost_for_prefix_tokens(prefix_tokens: float, model_name: str, gpu_name: str, gap_seconds: float = GAP_SECONDS) -> dict:
    model = get_model_info(model_name)
    gpu = get_gpu_info(gpu_name)
    prefix_kv_bytes = prefix_tokens * model['bytes_per_token']
    usable_kv = usable_kv_memory_bytes(model_name, gpu_name)
    gpu_fraction = prefix_kv_bytes / usable_kv
    storage_cost = gpu_fraction * gap_seconds * gpu['cost_per_second_usd']
    return {
        'prefix_kv_bytes': prefix_kv_bytes,
        'usable_kv_memory_bytes': usable_kv,
        'storage_gpu_fraction': gpu_fraction,
        'storage_cost': storage_cost,
    }


def compute_overall_prefix_curve(csv_path: str, target_session_id: str, model_name: str, gpu_name: str) -> dict:
    rows = load_session_rows(csv_path, target_session_id)
    if not rows:
        raise ValueError(f'No rows found for session_id={target_session_id!r}')

    gpu = get_gpu_info(gpu_name)
    first_prefix = rows[0]['prefix_length']
    cumulative_cost = compute_cost_from_tokens(
        first_prefix,
        gpu['throughput_tokens_per_second'],
        gpu['cost_per_second_usd'],
    )

    points = [
        {
            'turn': 0,
            'label': 'round0',
            'initial_prefix_tokens': first_prefix,
            'incremental_cost': cumulative_cost,
            'cumulative_cost': cumulative_cost,
        }
    ]

    for idx, row in enumerate(rows, start=1):
        extra_len = row['next_turn_added_length']
        storage = storage_cost_for_prefix_tokens(row['prefix_length'], model_name, gpu_name)
        recompute_new_cost = compute_cost_from_tokens(
            extra_len,
            gpu['throughput_tokens_per_second'],
            gpu['cost_per_second_usd'],
        )
        turn_cost = storage['storage_cost'] + recompute_new_cost
        cumulative_cost += turn_cost
        points.append(
            {
                'turn': idx,
                'label': row['turn_indices'],
                'prefix_tokens': row['prefix_length'],
                'new_tokens': extra_len,
                'storage_cost': storage['storage_cost'],
                'recompute_new_cost': recompute_new_cost,
                'prefix_kv_bytes': storage['prefix_kv_bytes'],
                'usable_kv_memory_bytes': storage['usable_kv_memory_bytes'],
                'storage_gpu_fraction': storage['storage_gpu_fraction'],
                'incremental_cost': turn_cost,
                'cumulative_cost': cumulative_cost,
            }
        )

    return {
        'session_id': target_session_id,
        'mode': 'overall_prefix',
        'metric': 'overall_cost',
        'model_name': model_name,
        'gpu_name': gpu_name,
        'gap_seconds': GAP_SECONDS,
        'points': points,
        'total_overall_cost': cumulative_cost,
    }


def compute_overall_compaction_curve(csv_path: str, target_session_id: str, compaction_fraction: float, model_name: str, gpu_name: str) -> dict:
    if not (0.0 <= compaction_fraction <= 1.0):
        raise ValueError('compaction_fraction must be between 0.0 and 1.0')

    rows = load_session_rows(csv_path, target_session_id)
    if not rows:
        raise ValueError(f'No rows found for session_id={target_session_id!r}')

    gpu = get_gpu_info(gpu_name)
    first_prefix = rows[0]['prefix_length']
    cumulative_cost = compute_cost_from_tokens(
        first_prefix,
        gpu['throughput_tokens_per_second'],
        gpu['cost_per_second_usd'],
    )

    points = [
        {
            'turn': 0,
            'label': 'round0',
            'initial_prefix_tokens': first_prefix,
            'incremental_cost': cumulative_cost,
            'cumulative_cost': cumulative_cost,
        }
    ]

    for idx, row in enumerate(rows, start=1):
        compacted_prefix = row['prefix_length'] * compaction_fraction
        extra_len = row['next_turn_added_length']
        recompute_prefix_cost = compute_cost_from_tokens(
            compacted_prefix,
            gpu['throughput_tokens_per_second'],
            gpu['cost_per_second_usd'],
        )
        recompute_new_cost = compute_cost_from_tokens(
            extra_len,
            gpu['throughput_tokens_per_second'],
            gpu['cost_per_second_usd'],
        )
        turn_cost = recompute_prefix_cost + recompute_new_cost
        cumulative_cost += turn_cost
        points.append(
            {
                'turn': idx,
                'label': row['turn_indices'],
                'compaction_fraction': compaction_fraction,
                'compacted_prefix_tokens': compacted_prefix,
                'new_tokens': extra_len,
                'recompute_prefix_cost': recompute_prefix_cost,
                'recompute_new_cost': recompute_new_cost,
                'incremental_cost': turn_cost,
                'cumulative_cost': cumulative_cost,
            }
        )

    return {
        'session_id': target_session_id,
        'mode': 'overall_compaction',
        'metric': 'overall_cost',
        'model_name': model_name,
        'gpu_name': gpu_name,
        'gap_seconds': GAP_SECONDS,
        'compaction_fraction': compaction_fraction,
        'points': points,
        'total_overall_cost': cumulative_cost,
    }
