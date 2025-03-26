import onnxruntime as ort
from transformers import AutoTokenizer
from fastapi import FastAPI
import numpy as np
from huggingface_hub import hf_hub_download

app = FastAPI()

MODEL_NAME = "nasar986/pModel"
ONNX_FILENAME = "model.onnx"  # Adjust if the file name is different

# Download the ONNX model from Hugging Face
onnx_model_path = hf_hub_download(repo_id=MODEL_NAME, filename=ONNX_FILENAME)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load ONNX model
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

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
