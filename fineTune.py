import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType  # LoRA integration

# 1. Configuration
MODEL_NAME = "gpt2"  # or "gpt2-medium"/"gpt2-large" for better results
OUTPUT_DIR = "./pModel"
DATASET_PATH = "training_data.json"

# 2. Load and prepare dataset
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def format_example(sample):
    return {"text": f"### Instruction: Answer based on resume\n### Input: {sample['input']}\n### Output: {sample['output']}"}

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(format_example)

def tokenize_function(samples):
    return tokenizer(
        samples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(["input", "output", "text"])

# 3. LoRA Configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"],  # Targeting attention layers
    bias="none"
)

# 4. Load and prepare model
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should show ~0.1% of parameters trainable

# 5. Optimized Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,  # Increased epochs
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Simulates larger batch size
    save_steps=500,
    logging_steps=100,
    learning_rate=3e-4,  # Higher LR for LoRA
    weight_decay=0.01,
    warmup_steps=500,  # Added warmup
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Auto-enable FP16 if GPU available
    optim="adamw_torch",  # Better optimizer
    report_to="none",
    push_to_hub=False,
    evaluation_strategy="no",
    lr_scheduler_type="cosine",  # Better learning rate schedule
)

# 6. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 8. Train and save
print("Starting training with LoRA...")
trainer.train()

# Save only the adapters to save space
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA adapters saved to {OUTPUT_DIR}")