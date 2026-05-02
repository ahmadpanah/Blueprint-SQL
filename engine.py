import sqlglot
from action_library import ACTION_SPACE
import re

class BlueprintEngine:
    def __init__(self, dialect="postgres"):
        self.dialect = dialect

    def parse_llm_blueprint(self, llm_output: str) -> list:
        """Parses the generated text into actionable primitives."""
        blueprint =[]
        # Regex to match: 1. ACTION_NAME(param1, param2)
        pattern = r"\d+\.\s*([A-Z_]+)\((.*?)\)"
        matches = re.findall(pattern, llm_output)
        
        for action_name, param_str in matches:
            if action_name in ACTION_SPACE:
                params = [p.strip() for p in param_str.split(",")] if param_str else[]
                blueprint.append({"action": action_name, "params": params})
        return blueprint

    def execute_blueprint(self, original_sql: str, blueprint: list) -> str:
        """Deterministically applies the blueprint to the SQL AST."""
        try:
            # Step 1: Parse original SQL into AST
            ast = sqlglot.parse_one(original_sql, read=self.dialect)
            
            # Step 2: Sequentially apply actions
            for step in blueprint:
                action_handler = ACTION_SPACE[step["action"]]
                ast = action_handler.apply(ast, step["params"])
                
            # Step 3: Unparse back to SQL
            rewritten_sql = ast.sql(dialect=self.dialect)
            return rewritten_sql
            
        except Exception as e:
            # Robustness Guarantee: If execution fails, fallback to original query
            return original_sql