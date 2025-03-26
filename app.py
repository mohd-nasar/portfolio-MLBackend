from fastapi import FastAPI
from transformers import pipeline  # No direct torch import

app = FastAPI()

# 1. Load model with ONNX runtime (lower memory)
generator = pipeline(
    "text-generation",
    model="nasar986/pModel",  # Your fine-tuned model
    device=-1,  # CPU only
    framework="pt",  # Still uses PyTorch but more efficiently
    torch_dtype="auto"  # Automatic precision
)

# 2. Minimal generation function
def generate_text(prompt: str):
    return generator(
        prompt,
        max_length=100,  # Reduced from 150
        num_return_sequences=1,
        truncation=True
    )[0]['generated_text']

# 3. FastAPI endpoint
@app.post("/generate")
async def generate_response(payload: dict):
    prompt = payload.get("prompt", "")
    if not prompt or len(prompt) > 500:
        return {"error": "Prompt must be 1-500 characters"}
    
    try:
        return {"output": generate_text(prompt)[:500]}  # Hard cap output
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)  # Single worker
