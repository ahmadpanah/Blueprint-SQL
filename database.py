import psycopg2
import json
from config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

class DatabaseManager:
    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            self.cursor = self.conn.cursor()
            print("✅ Database connection established.")
        except psycopg2.OperationalError as e:
            print(f"❌ Could not connect to database: {e}")
            self.conn = None
            self.cursor = None

    def execute_query(self, query):
        if not self.cursor: return None, "No DB connection"
        try:
            self.cursor.execute(query)
            return self.cursor.fetchall(), None
        except Exception as e:
            self.conn.rollback()
            return None, str(e)

    def get_query_cost(self, query):
        if not self.cursor: return float('inf'), "No DB connection"
        try:
            # Using EXPLAIN with JSON format provides a structured way to get the cost
            self.cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
            explain_plan = self.cursor.fetchone()[0]
            # The total cost is at the top of the plan
            return float(explain_plan[0]['Plan']['Total Cost']), None
        except Exception as e:
            self.conn.rollback()
            return float('inf'), str(e)

    def check_equivalence(self, query1, query2):
        """A practical, if not formal, way to check equivalence."""
        if not self.cursor: return False, "No DB connection"
        
        res1, err1 = self.execute_query(query1)
        res2, err2 = self.execute_query(query2)

        if err1 or err2:
            return False, f"Execution error during equivalence check. Q1: {err1}, Q2: {err2}"

        # Sort results to handle differences in row ordering
        # This requires results to be comparable (e.g., tuples of primitives)
        try:
            sorted_res1 = sorted(res1)
            sorted_res2 = sorted(res2)
            return sorted_res1 == sorted_res2, None
        except TypeError:
            # Fallback for non-sortable results: compare multisets
            return len(res1) == len(res2) and all(res1.count(row) == res2.count(row) for row in res1), None

    def close(self):
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print("⚪️ Database connection closed.")