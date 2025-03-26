import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel
from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

# Load fine-tuned model
MODEL_PATH = "nasar986/pModel"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained("gpt2")  # Base GPT-2
model = PeftModel.from_pretrained(model, MODEL_PATH)  # Load LoRA adapters
model.eval()

# Function to generate text
def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Formatting response for a more realistic output
    response_text = generated_text.strip().replace("\n", " ").replace("###", "").strip()
    return response_text

# FastAPI endpoint
@app.post("/generate")
async def generate_response(payload: dict):
    prompt = payload.get("prompt", "")
    if not prompt:
        return {"error": "Prompt is required"}
    response_text = generate_text(prompt)
    return {"output": response_text}

# Run locally for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
