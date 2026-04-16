from __future__ import annotations

GPU_CATALOG = {
    '1x H100 80GB': {
        'display_name': '1x H100 80GB',
        'throughput_tokens_per_second': 3621.0,
        'hourly_cost_usd': 2.39,
        'total_memory_bytes': 80 * 10**9,
        'gpu_memory_utilization': 0.90,
        'reserved_runtime_bytes': 2 * 1024**3,
    },
    '1x A100 80GB': {
        'display_name': '1x A100 80GB',
        'throughput_tokens_per_second': 2200.0,
        'hourly_cost_usd': 1.64,
        'total_memory_bytes': 80 * 10**9,
        'gpu_memory_utilization': 0.90,
        'reserved_runtime_bytes': 2 * 1024**3,
    },
    '1x L40S 48GB': {
        'display_name': '1x L40S 48GB',
        'throughput_tokens_per_second': 1200.0,
        'hourly_cost_usd': 0.99,
        'total_memory_bytes': 48 * 10**9,
        'gpu_memory_utilization': 0.90,
        'reserved_runtime_bytes': 2 * 1024**3,
    },
}

DEFAULT_GPU = '1x H100 80GB'


def get_gpu_info(gpu_name: str) -> dict:
    if gpu_name not in GPU_CATALOG:
        raise ValueError(f'Unsupported GPU: {gpu_name}')
    info = dict(GPU_CATALOG[gpu_name])
    info['name'] = gpu_name
    info['cost_per_second_usd'] = info['hourly_cost_usd'] / 3600.0
    return info


def gpu_options() -> list[dict]:
    return [
        {
            'name': name,
            'display_name': data['display_name'],
            'throughput_tokens_per_second': data['throughput_tokens_per_second'],
            'hourly_cost_usd': data['hourly_cost_usd'],
        }
        for name, data in GPU_CATALOG.items()
    ]
