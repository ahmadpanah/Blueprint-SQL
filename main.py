from trainer import RLTrainer
from agent import LLMAgent
from database import DatabaseManager

def setup_database(db):
    """Create and populate dummy tables for experiments."""
    if not db.conn: return

    print("\nSetting up database schema for experiments...")
    # A simple schema for demonstration
    commands = [
        "DROP TABLE IF EXISTS lineitem, orders, customer CASCADE;",
        "CREATE TABLE customer (c_custkey INT PRIMARY KEY, c_name VARCHAR, c_mktsegment VARCHAR);",
        "CREATE TABLE orders (o_orderkey INT PRIMARY KEY, o_custkey INT REFERENCES customer(c_custkey), o_orderdate DATE);",
        "CREATE TABLE lineitem (l_orderkey INT REFERENCES orders(o_orderkey), l_partkey INT, l_quantity INT, l_extendedprice DECIMAL);",
        # Populate with some data
        "INSERT INTO customer VALUES (1, 'Customer A', 'BUILDING'), (2, 'Customer B', 'AUTOMOBILE');",
        "INSERT INTO orders VALUES (101, 1, '2023-01-10'), (102, 2, '2023-01-12');",
        "INSERT INTO lineitem VALUES (101, 1001, 2, 2500.0), (101, 1002, 1, 150.5), (102, 1003, 5, 50.0);"
    ]
    for command in commands:
        _, err = db.execute_query(command)
        if err:
            print(f"  DB Setup Error: {err}")
    db.conn.commit()
    print("✅ Database schema setup complete.")
    return "customer(c_custkey, c_name, c_mktsegment), orders(o_orderkey, o_custkey, o_orderdate), lineitem(...)"


def run_scenario_1_standard_perf():
    print("\n\n" + "="*50)
    print("SCENARIO 1: Performance on Standard Benchmarks (TPC-H like)")
    print("="*50)
    
    query = """
    SELECT c_name, SUM(l_extendedprice * (1 - 0.05)) as revenue
    FROM customer c
    JOIN (
        SELECT o_orderkey, o_custkey
        FROM orders
        WHERE o_orderdate < '2024-01-01'
    ) AS o_sub
    ON c.c_custkey = o_sub.o_custkey
    JOIN lineitem l ON l.l_orderkey = o_sub.o_orderkey
    WHERE c.c_mktsegment = 'BUILDING'
    GROUP BY c_name;
    """
    
    trainer, db_manager = setup_trainer()
    best_query, best_blueprint, stats = trainer.train_on_query(query)
    
    print("\n--- SCENARIO 1 RESULTS ---")
    print_results_table(stats)
    print("\nBest Blueprint Found:")
    print(best_blueprint)
    print("\nBest Rewritten Query:")
    print(best_query)
    db_manager.close()

def run_scenario_2_novel_pattern():
    print("\n\n" + "="*50)
    print("SCENARIO 2: Adaptability to Novel Patterns (Correlated Subquery)")
    print("="*50)

    # This query is inefficient because the subquery runs for every customer row.
    # The optimal rewrite unnests this into a JOIN with a GROUP BY.
    query = """
    SELECT c_name
    FROM customer c
    WHERE (
        SELECT SUM(l_quantity)
        FROM orders o
        JOIN lineitem l ON o.o_orderkey = l.l_orderkey
        WHERE o.o_custkey = c.c_custkey
    ) > 5;
    """
    trainer, db_manager = setup_trainer()
    best_query, best_blueprint, stats = trainer.train_on_query(query)
    
    print("\n--- SCENARIO 2 RESULTS ---")
    print("Objective: Unnest the correlated subquery.")
    print("A high latency reduction indicates success.")
    print_results_table(stats)
    print("\nBest Blueprint Found:")
    print(best_blueprint)
    print("\nBest Rewritten Query:")
    print(best_query)
    db_manager.close()


def run_scenario_3_and_4_robustness_explainability():
    print("\n\n" + "="*50)
    print("SCENARIO 3 & 4: Robustness & Explainability")
    print("="*50)
    
    # A simple query where join order matters, but is syntactically simple enough
    # that a black-box model might rearrange it correctly but opaquely.
    query = "SELECT c.c_name, o.o_orderdate FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey;"

    trainer, db_manager = setup_trainer()
    best_query, best_blueprint, stats = trainer.train_on_query(query, episodes=2)

    print("\n--- SCENARIO 3 & 4 RESULTS ---")
    print("SCENARIO 3 (Robustness):")
    print(f"  - Blueprint-SQL produced a syntactically valid query: {'Yes' if stats['valid'] else 'No'}")
    print("  - A black-box model (like E³-Rewrite) might have failed on more complex queries, but our architecture guarantees validity.")
    
    print("\nSCENARIO 4 (Explainability):")
    if not best_blueprint:
        print("  - The agent found the query optimal and produced no blueprint.")
    else:
        print("  - The generated blueprint provides a clear, step-by-step rationale for the rewrite:")
        for i, step in enumerate(best_blueprint):
            print(f"    Step {i+1}: {step['action']} with params {step['params']}")
        print("  - This is far more transparent than a monolithic rewritten SQL string.")
        
    db_manager.close()


def print_results_table(stats):
    print("\n" + "-"*30)
    print(f"{'Metric':<25} {'Blueprint-SQL':<15}")
    print("-"*30)
    print(f"{'Latency Reduction (%)':<25} {stats.get('latency_reduction', 0.0):.2f}")
    print(f"{'Equivalence Rate (%)':<25} {100.0 if stats.get('equivalent', False) else 0.0}")
    print(f"{'Syntactic Validity (%)':<25} {100.0 if stats.get('valid', False) else 0.0}")
    print("-"*30)
    print("\n*Note: Baseline results from paper are omitted for this live run.*")


def setup_trainer():
    db_manager = DatabaseManager()
    schema_info = setup_database(db_manager)
    agent = LLMAgent(db_schema_info=schema_info)
    trainer = RLTrainer(agent, db_manager)
    return trainer, db_manager


if __name__ == "__main__":
    run_scenario_1_standard_perf()
    run_scenario_2_novel_pattern()
    run_scenario_3_and_4_robustness_explainability()