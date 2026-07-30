"""Output-field filtering helpers.

Used by main.py and execute_user_sweep_config.py to keep printed JSON compact while
preserving every field in the actual exported files.
"""

from typing import Any


# Fields that are rarely useful when pasted into a console/terminal table but
# remain available in JSON files and webserver responses.
_VERBOSE_FIELDS: set[str] = {
    "scenario_name",
    "total_gpus",
    "num_prefill_workers",
    "num_decode_workers",
    "prefill_gpus_per_worker",
    "decode_gpus_per_worker",
    "memory_gb",
    "ram_cache_usage_bytes",
    "ssd_cache_usage_bytes",
    "s3_cache_usage_bytes",
    "s3_peak_cache_usage_bytes",
    "ram_cache_capacity_bytes",
    "ssd_cache_capacity_bytes",
    "per_request_stats",
    "avg_prefill_wait_ms",
    "max_prefill_wait_ms",
    "avg_prefill_download_active_ms",
    "avg_prefill_download_wait_ms",
    "avg_prefill_upload_active_ms",
    "avg_prefill_upload_wait_ms",
    "avg_decode_wait_ms",
    "max_decode_wait_ms",
    "avg_decode_download_active_ms",
    "avg_decode_download_wait_ms",
    "avg_decode_upload_active_ms",
    "avg_decode_upload_wait_ms",
    "max_clean_ttft_ms",
    "avg_clean_latency_ms",
    "max_clean_latency_ms",
    "router_active_work_scale",
    "router_device_credit",
}


# Core fields shown in the compact table view.
_DEFAULT_FIELDS: list[str] = [
    "label",
    "ttft",
    "tpot",
    "kv_download_time",
    "kv_upload_time",
    "request_latency",
    "max_request_latency",
    "max_ttft",
    "max_tpot",
    "tokens_per_second",
    "tokens_per_second_per_gpu",
    "tokens_per_second_per_user",
    "seq_per_second",
    "concurrency",
    "avg_prefill_time_ms",
    "avg_decode_time_ms",
    "avg_clean_ttft_ms",
    "compute_price_usd_per_hour",
    "s3_cost_usd_per_hour",
    "s3_storage_cost_usd_per_hour",
    "total_cost_usd_per_hour",
    "router_active_work_scale",
    "router_device_credit",
]


def filter_dict(
    data: dict[str, Any],
    allow: list[str] | None = None,
    deny: set[str] | None = None,
) -> dict[str, Any]:
    """Return a shallow copy of ``data`` keeping only selected fields.

    Parameters
    ----------
    allow:
        Whitelist of keys to retain. Defaults to ``_DEFAULT_FIELDS``.
    deny:
        Optional denylist applied after ``allow``.
    """
    if allow is None:
        allow = _DEFAULT_FIELDS
    result = {k: data[k] for k in allow if k in data}
    if deny:
        for k in list(result):
            if k in deny:
                result.pop(k)
    return result


def compact_result(data: dict[str, Any]) -> dict[str, Any]:
    """Default compact view of a SimulationResult.to_dict() row."""
    return filter_dict(data, allow=_DEFAULT_FIELDS)


def compact_json(data: dict[str, Any], indent: int = 2) -> str:
    """Return a compact JSON string for the selected fields."""
    import json

    return json.dumps(compact_result(data), indent=indent)
