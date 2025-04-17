import uvicorn
from api.predict_api import app  # Import the FastAPI app from predict_api.py

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
