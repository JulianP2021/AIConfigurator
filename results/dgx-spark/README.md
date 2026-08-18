# Time calculation
uv run scripts/prefill_decode_time.py --isl 25000 --chunked 2048 --flops 213e12 --mem-bw 273e9


# DGX SPARK
The Dockerfile is from the cloned repo https://github.com/eugr/spark-vllm-docker, with the addition of a LMCache download(unnecessary for this benchmark):


## Install LMCache for CUDA 13. Avoids pulling the wrong nixl-cu12 variant.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install lmcache && \
    uv pip uninstall -y nixl-cu12 || true


## vLLM bench command
vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --model Qwen/Qwen3.8-27B \
    --dataset-name random \
    --random-input-len 25000 \
    --random-output-len 100 \
    --ignore-eos \
    --num-prompts 5 \
    --max-concurrency 1 \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "50,90,95,99" \
    --save-result
'