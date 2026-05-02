import psycopg2
import time
import hashlib

class DatabaseEnv:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.conn.autocommit = True
        
    def get_query_cost(self, sql: str) -> float:
        """Returns PostgreSQL EXPLAIN cost."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"EXPLAIN {sql}")
                plan = cur.fetchone()[0]
                # Extract cost from "cost=0.00..1234.56"
                cost = float(plan.split("cost=")[1].split("..")[1].split(" ")[0])
                return cost
        except Exception:
            return float('inf')

    def check_equivalence(self, orig_sql: str, rewritten_sql: str) -> bool:
        """Executes both queries with LIMIT 100 and hashes results to check semantic equivalence."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT * FROM ({orig_sql}) AS t LIMIT 100")
                orig_res = str(cur.fetchall()).encode()
                
                cur.execute(f"SELECT * FROM ({rewritten_sql}) AS t LIMIT 100")
                new_res = str(cur.fetchall()).encode()
                
                return hashlib.md5(orig_res).hexdigest() == hashlib.md5(new_res).hexdigest()
        except Exception:
            return False

    def compute_reward(self, original_sql: str, rewritten_sql: str) -> float:
        """
        Implements Equation 2 from the paper:
        R(B) = lambda_perf * R_perf + lambda_equiv * R_equiv + lambda_exec * R_exec
        """
        lambda_perf, lambda_equiv, lambda_exec = 1.0, 5.0, 2.0
        
        orig_cost = self.get_query_cost(original_sql)
        new_cost = self.get_query_cost(rewritten_sql)
        
        # Executability Reward (R_exec)
        if new_cost == float('inf'):
            return -lambda_exec
            
        # Equivalence Reward (R_equiv)
        if not self.check_equivalence(original_sql, rewritten_sql):
            return -lambda_equiv
            
        # Performance Reward (R_perf)
        r_perf = max(0, (orig_cost - new_cost) / orig_cost)
        
        return lambda_perf * r_perf