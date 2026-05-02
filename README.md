# Blueprint-SQL: Explainable Global Query Optimization via LLM-Driven Strategic Restructuring

This repository contains the official, full-scale implementation of Blueprint-SQL. It trains a Large Language Model (CodeLlama-13B) via Proximal Policy Optimization (PPO) to generate human-readable optimization "blueprints" rather than opaque SQL strings.

## Repository Structure
- `action_library.py`: Implementation of the 10-action AST manipulation library using `sqlglot`.
- `engine.py`: The deterministic Blueprint Execution Engine that guarantees 100% syntactic validity.
- `env.py`: Database connection, execution, and multi-objective reward calculation.
- `train_ppo.py`: The complete RL fine-tuning pipeline using `trl` and `peft` (LoRA).
- `config.yaml`: Hyperparameters for training and database connection.

## Setup Instructions
1. Install requirements:
   ```bash
   pip install torch transformers trl peft sqlglot psycopg2-binary pyyaml
   ```
2. Configure your PostgreSQL database credentials in `config.yaml`.
3. Load the TPC-H and Novel-Q schemas into your target Postgres instance.
4. Run the PPO training pipeline:
   ```bash
   python train_ppo.py --config config.yaml
   ```