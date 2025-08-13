import sqlglot
from actions import ACTION_MAPPING

class BlueprintExecutor:
    def execute(self, original_query, blueprint):
        """
        Applies a blueprint to an original query.
        Returns the rewritten query string and any errors.
        """
        try:
            # 1. Parse the original query into an AST
            ast = sqlglot.parse_one(original_query)
        except Exception as e:
            return None, f"AST Parsing Error: {e}"

        # 2. Sequentially apply each action in the blueprint
        for action_item in blueprint:
            action_name = action_item.get("action")
            params = action_item.get("params", {})

            if action_name in ACTION_MAPPING:
                action_func = ACTION_MAPPING[action_name]
                try:
                    # Apply the function to the current AST
                    ast = action_func(ast, **params)
                except Exception as e:
                    return None, f"Error applying action '{action_name}': {e}"
            else:
                return None, f"Unknown action: {action_name}"

        # 3. Unparse the final AST back into a SQL string
        return ast.sql(pretty=True), None