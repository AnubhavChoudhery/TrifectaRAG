from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
from trifecta import TrifectaClient

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Trifecta Client
# We use the local TrifectaClient which wraps the C++ engine
client = TrifectaClient(device="cpu")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    last_user_message = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    
    if not last_user_message:
        return {"role": "assistant", "content": "I didn't receive a message."}

    try:
        # 1. Query the local Trifecta engine (C++ HNSW + CLIP + KG)
        # This returns raw search results from the vector/lexical index
        raw_results = client.query(text=last_user_message, top_k=5)
        results = client.get_results(raw_results)
        
        # 2. Extract context and citations
        context_parts = []
        citations = []
        for i, res in enumerate(results):
            text = res.get("text", "")
            source = res.get("metadata", {}).get("source", "Unknown")
            page = res.get("metadata", {}).get("page", "?")
            
            context_parts.append(f"[Source {i+1}: {source}, Page {page}]\n{text}")
            citations.append({
                "id": i + 1,
                "source": f"{source} (Page {page})",
                "content": text[:300] + "..." if len(text) > 300 else text,
                "type": res.get("type", "text")
            })

        # 3. For a purely local version without Anthropic, we can return the top result 
        # or a compiled list of excerpts. 
        # If you WANT to use the local C++ logic for the final answer, 
        # the engine provides the retrieval, but typically a small local LLM performs the synthesis.
        
        if not results:
            return {
                "role": "assistant",
                "content": "I couldn't find any relevant information in the indexed documents.",
                "citations": []
            }

        # Synthesize a response locally from the retrieved chunks
        response_text = "Based on the internal documents, I found the following information:\n\n"
        response_text += "\n\n".join([f"**From {c['source']}**: {c['content']}" for c in citations[:2]])
        
        return {
            "role": "assistant",
            "content": f"I found some information in the local index regarding '{last_user_message}':\n\n{results[0]['text']}\n\nYou can see more details in the citations below.",
            "citations": citations
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
