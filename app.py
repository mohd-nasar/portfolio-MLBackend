import onnxruntime as ort
from transformers import AutoTokenizer
from fastapi import FastAPI
import numpy as np

app = FastAPI()

MODEL_NAME = "nasar986/pModel"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
session = ort.InferenceSession(f"{MODEL_NAME}.onnx", providers=["CPUExecutionProvider"])

@app.post("/generate")
async def generate_response(payload: dict):
    prompt = payload.get("prompt", "")
    if not prompt:
        return {"error": "Prompt is required"}
    
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    output = session.run(None, {"input_ids": input_ids})[0]
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {"output": response_text.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
