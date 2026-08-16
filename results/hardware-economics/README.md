# Generate the focussed machines from AWS p4de.24xlarge (A100_80GB x8) base machine
uv run scripts/generate_focused_machines.py A100_80GB --num-gpus 8 --clean --write --pcie-bw-gbps 250 --ram-mem-gb 1200 --nvlink-bw-gbps 350 --ssd-mem-gb 1080 --inet-bw-gbps 50 --inter-node-bw-gbps 100


# Create configs
uv run configs/create_hardware_economics_config.py --config-name config-hardware.json --config-type colocated --prefill-gpus-per-node 4 --custom-hardware src/hardware/data/custom_hardware.json --prefill-nodes 5 --decode-nodes 5


# Simulate configs
uv run configs/execute_hardware_economics_config.py --config config-hardware.json --results-dir results-hardware-economics --ttft-values 12,15,17 --user-delay-values 60,600 --seeds 3
