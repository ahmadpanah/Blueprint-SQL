# Blueprint-SQL: A Proof-of-Concept Implementation

This repository provides a proof-of-concept implementation of the "From Black Box to Blueprint: Generating Composable and Explainable SQL Rewrite Strategies" paper. It demonstrates the core idea of using a Large Language Model (LLM) to generate a high-level "blueprint" of rewrite actions, which are then deterministically applied to a query's Abstract Syntax Tree (AST).

## Core Features

- **Blueprint Generation**: Uses an LLM (via OpenAI's API) to generate a sequence of rewrite actions.
- **AST Manipulation**: Leverages the `sqlglot` library to parse SQL, apply transformations, and unparse back to SQL.
- **Blueprint Execution**: A deterministic engine that applies the generated blueprint, guaranteeing syntactic correctness.
- **Simplified RL Loop**: A training loop that simulates reinforcement learning by rewarding the LLM for good blueprints and using them as few-shot examples in subsequent prompts.
- **Live Database Evaluation**: Connects to a real PostgreSQL database to execute queries, estimate costs using `EXPLAIN`, and check for result equivalence.

## Project Structure

- `main.py`: The main entry point to run the experimental scenarios described in the paper.
- `trainer.py`: Orchestrates the RL training loop for a given query.
- `agent.py`: The LLM agent responsible for generating blueprints.
- `blueprint.py`: The `BlueprintExecutor` that applies a blueprint to a query.
- `actions.py`: Defines the composable rewrite action functions (e.g., `CONVERT_SUBQUERY_TO_CTE`).
- `database.py`: Manages the connection and all interactions with the PostgreSQL database.
- `config.py`: Loads configuration from the `.env` file.

## Setup Instructions

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/ahmadpanah/Blueprint-SQL
    cd blueprint-sql
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up PostgreSQL Database**
    - Make sure you have PostgreSQL installed and running.
    - Create a new database for the experiments (e.g., `blueprint_db`).
    - The script will automatically create and populate the necessary tables.

4.  **Configure Environment Variables**
    - Create a file named `.env` in the root of the project.
    - Add your credentials to this file. See `config.py` for the required variables.
      ```
      OPENAI_API_KEY="your_openai_api_key"
      DB_NAME="blueprint_db"
      DB_USER="your_postgres_user"
      DB_PASSWORD="your_postgres_password"
      DB_HOST="localhost"
      DB_PORT="5432"
      ```

## How to Run

Execute the main script from your terminal. It will run through the evaluation scenarios sequentially.

```bash
python main.py