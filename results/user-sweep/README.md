# Creating configs
uv run configs/create_user_sweep_config.py --config-name config-users.json --high-end-only

# Simulate configs
uv run configs/execute_user_sweep_config.py --config config-users.json --results-dir results-user-sweep --users 100,150,200