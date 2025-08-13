import openai
import json
from config import OPENAI_API_KEY, LLM_MODEL

class LLMAgent:
    def __init__(self, db_schema_info=""):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        openai.api_key = OPENAI_API_KEY
        self.db_schema_info = db_schema_info
        self.few_shot_examples = []

    def generate_blueprint(self, query):
        system_prompt = self._build_system_prompt()
        user_prompt = f"""
        Given the following SQL query, generate an optimal rewrite blueprint.
        
        QUERY:
        ```sql
        {query}
        ```
        """
        
        try:
            response = openai.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            blueprint_json = response.choices[0].message.content
            # The API should return a JSON object, so we parse it
            data = json.loads(blueprint_json)
            return data.get("blueprint", []), None
        except Exception as e:
            return None, f"LLM API Error: {e}"

    def add_successful_example(self, query, blueprint, reward):
        """Adds a successful example to be used for in-context learning."""
        example = {
            "query": query,
            "blueprint": blueprint,
            "reward": reward
        }
        self.few_shot_examples.append(example)
        # Keep only the last few best examples to fit in the context window
        self.few_shot_examples = sorted(self.few_shot_examples, key=lambda x: x['reward'], reverse=True)[:3]

    def _build_system_prompt(self):
        prompt = """
You are Blueprint-SQL, an expert database query optimizer. Your task is to generate a 'rewrite blueprint' to optimize a given SQL query. A blueprint is a JSON array of actions.

You must only output a valid JSON object with a single key: "blueprint".

AVAILABLE ACTIONS:
1.  `CONVERT_SUBQUERY_TO_CTE`: Hoists a subquery into a Common Table Expression.
    - `params`: {"subquery_alias": "alias_of_the_subquery", "cte_name": "new_cte_name"}
2.  `REORDER_JOIN`: Reorders two tables in a join. (Note: May not always apply).
    - `params`: {"table1_name": "name_of_first_table", "table2_name": "name_of_second_table"}

You can also return an empty blueprint `{"blueprint": []}` if you believe the query is already optimal.

"""
        if self.db_schema_info:
            prompt += f"DATABASE SCHEMA:\n{self.db_schema_info}\n\n"
        
        if self.few_shot_examples:
            prompt += "Here are some examples of successful blueprints:\n"
            for ex in self.few_shot_examples:
                prompt += f"--- Example ---\nQuery: {ex['query']}\nBlueprint: {json.dumps(ex['blueprint'])}\n"
        
        return prompt