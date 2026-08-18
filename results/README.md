# KV break even
uv run python scripts/kv_break_even.py --model Qwen/Qwen3.8-27B --isl 200000 --machine-hardware "AWS p4de.24xlarge (A100 80GB x8)"

# Gemini storage tier
uv run scripts/kv_storage_cost.py --tokens 1000000 --kv-size-gb-per-token 0.00032


# Router parameter results
uv run main.py --prefill-hardware "Focused A100_80GB x8 r1200 s1080 p250 nvl350 sbw2.1 in100.0/100.0 inet50.0/50.0" --num-prefill-nodes 5 --prefill-gpus-per-node 4 --sla '{"ttft_ms":15000,"tpot_ms":70}' --router-active-work-scale X --users Y

# Other results
Other results are explained in the subfolders