1. Install dependencies
2. pip install fastapi uvicorn transformers torch

Run the FastAPI server
uvicorn main:app --reload

Example Request
POST to /chat with JSON:
{
  "message": "Hello, how are you?"
}

Response:

{
  "response": "I'm doing great! How can I help you today?"
}


Notes:
chat_history_ids is global, meaning conversations persist between API calls. You can reset or isolate per user if needed.

For a real-world app, manage conversation history per session or user ID.

