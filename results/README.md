# KV break even
uv run python scripts/kv_break_even.py --model Qwen/Qwen3.8-27B --isl 200000 --machine-hardware "AWS p4de.24xlarge (A100 80GB x8)"

# Gemini storage tier
uv run scripts/kv_storage_cost.py --tokens 1000000 --kv-size-gb-per-token 0.00032


# Router parameter results
uv run main.py --prefill-hardware "Focused H200 x8 r2200 s4080 p504 nvl450 sbw8.0 in100.0/100.0 inet25.0/25.0" --num-prefill-nodes 5 --prefill-gpus-per-node 4 --sla '{"ttft_ms":15000,"tpot_ms":300}' --router-active-work-scale X

# Other results
Other results are explained in the subfolders