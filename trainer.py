from database import DatabaseManager
from blueprint import BlueprintExecutor
from agent import LLMAgent

class RLTrainer:
    def __init__(self, agent: LLMAgent, db_manager: DatabaseManager):
        self.agent = agent
        self.db = db_manager
        self.executor = BlueprintExecutor()

    def train_on_query(self, query, episodes=5):
        """
        Runs a simplified RL loop for a single query.
        Returns the best query and its stats.
        """
        original_cost, err = self.db.get_query_cost(query)
        if err:
            return query, {"error": f"Could not get cost of original query: {err}"}

        print(f"\n--- Training on Query ---")
        print(f"Original Query Cost: {original_cost:.2f}")

        best_blueprint = []
        best_rewritten_query = query
        best_cost = original_cost
        best_stats = {
            "cost": original_cost,
            "latency_reduction": 0.0,
            "equivalent": True,
            "valid": True
        }

        for i in range(episodes):
            print(f"\n--- Episode {i+1}/{episodes} ---")
            
            # 1. Agent generates a blueprint
            blueprint, err = self.agent.generate_blueprint(query)
            if err:
                print(f"  Agent error: {err}")
                continue
            print(f"  Generated Blueprint: {blueprint}")
            
            if not blueprint:
                print("  Agent suggested no changes.")
                continue

            # 2. Executor applies the blueprint
            rewritten_query, err = self.executor.execute(query, blueprint)
            if err:
                print(f"  Executor error: {err}")
                continue
            
            # 3. Evaluate the new query
            new_cost, err = self.db.get_query_cost(rewritten_query)
            valid = err is None
            
            if not valid:
                print(f"  Rewritten query is invalid: {err}")
                reward = -1.0 # Large penalty for invalidity
            else:
                equivalent, eq_err = self.db.check_equivalence(query, rewritten_query)
                if not equivalent:
                    print(f"  Rewritten query is NOT equivalent. {eq_err or ''}")
                    reward = -1.0 # Large penalty for non-equivalence
                else:
                    print(f"  Equivalent. New Cost: {new_cost:.2f}")
                    # Reward is proportional to cost reduction
                    reward = (original_cost - new_cost) / original_cost if original_cost > 0 else 0
                    
                    if new_cost < best_cost:
                        print(f"  Found new best query! Cost reduction: {reward*100:.2f}%")
                        best_cost = new_cost
                        best_blueprint = blueprint
                        best_rewritten_query = rewritten_query
                        best_stats = {
                            "cost": new_cost,
                            "latency_reduction": reward * 100,
                            "equivalent": True,
                            "valid": True
                        }
            
            # 4. "Learn" by adding good examples to the agent's context
            if reward > 0.1: # Only learn from significant improvements
                print(f"  Learning from successful blueprint (Reward: {reward:.2f})")
                self.agent.add_successful_example(query, blueprint, reward)

        print("\n--- Training Finished ---")
        return best_rewritten_query, best_blueprint, best_stats