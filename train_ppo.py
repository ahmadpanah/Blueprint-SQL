import torch
import yaml
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from engine import BlueprintEngine
from env import DatabaseEnv
import pandas as pd

def load_training_data(filepath="data/novel_q_train.csv"):
    df = pd.read_csv(filepath)
    return df['sql'].tolist(), df['db_stats_prompt'].tolist()

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Initialize Database Environment and Execution Engine
    db_env = DatabaseEnv(config["database"])
    engine = BlueprintEngine()

    # 2. Setup CodeLlama-13B with LoRA for efficient RL fine-tuning
    model_name = "meta-llama/CodeLlama-13b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name, load_in_4bit=True, device_map="auto"
    )
    model.pretrained_model = get_peft_model(model.pretrained_model, lora_config)

    # 3. Setup PPO Trainer
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=8,
        mini_batch_size=2,
        gradient_accumulation_steps=4
    )
    ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

    # 4. Load Data (Novel-Q & TPC-H)
    queries, stat_prompts = load_training_data()

    print("Starting Blueprint-SQL PPO Training...")
    epochs = config["training"]["epochs"]
    
    # 5. RL Training Loop
    for epoch in range(epochs):
        for i, (orig_sql, stats) in enumerate(zip(queries, stat_prompts)):
            
            # Construct State Representation (s_t) with DB Stats mapping
            prompt = f"[SCHEMA & STATS]\n{stats}\n[CURRENT AST]\n{orig_sql}\nGenerate Blueprint:\n"
            input_tensors = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            # Agent takes action (generates blueprint)
            response_tensors = ppo_trainer.generate(
                input_tensors, max_new_tokens=128, do_sample=True, top_p=0.9
            )
            llm_output = tokenizer.decode(response_tensors[0][input_tensors.shape[1]:])

            # Deterministic Execution (AST Manipulation)
            blueprint = engine.parse_llm_blueprint(llm_output)
            rewritten_sql = engine.execute_blueprint(orig_sql, blueprint)

            # Receive Reward from Database
            reward_val = db_env.compute_reward(orig_sql, rewritten_sql)
            reward_tensor = torch.tensor([reward_val], dtype=torch.float32).to(model.device)

            # PPO Policy Update
            stats = ppo_trainer.step([input_tensors[0]],[response_tensors[0][input_tensors.shape[1]:]], [reward_tensor[0]])
            
            print(f"Epoch {epoch} | Query {i} | Reward: {reward_val:.4f} | Blueprint: {blueprint}")

    # Save finalized policy
    model.save_pretrained("blueprint_sql_codellama")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()