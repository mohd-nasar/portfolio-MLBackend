
from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def generate_response():
    
    return {"output": "API IS RunninG"}

# Run locally for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
