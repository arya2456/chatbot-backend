from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from pinecone import Pinecone

app = FastAPI()

# Allow dashboard to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(host=os.environ.get("PINECONE_HOST"))

class TrainRequest(BaseModel):
    url: str
    gemini_key: str

class ChatRequest(BaseModel):
    message: str
    url: str
    gemini_key: str

def get_clean_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts/styles
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
            
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit text to 8000 characters to prevent hitting size limits
        return text[:8000] 
    except Exception as e:
        return ""

@app.get("/")
def home():
    return {"status": "Chatbot Brain is Active"}

@app.post("/train")
def train_bot(req: TrainRequest):
    try:
        # 1. Scrape
        website_content = get_clean_text(req.url)
        if not website_content:
            return {"status": "error", "message": "Could not read website content"}

        # 2. Setup Gemini
        genai.configure(api_key=req.gemini_key)
        
        # 3. Create Embedding (USING NEW MODEL)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=website_content,
            task_type="retrieval_document",
            title="Website Context"
        )
        
        # 4. Store in Pinecone
        vector = result['embedding']
        index.upsert(
            vectors=[
                {
                    "id": "page_content", 
                    "values": vector, 
                    "metadata": {"text": website_content}
                }
            ],
            namespace=req.url
        )
        return {"status": "success", "message": "Website learned!"}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_bot(req: ChatRequest):
    try:
        genai.configure(api_key=req.gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Search Pinecone (USING NEW MODEL)
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=req.message,
            task_type="retrieval_query"
        )['embedding']
        
        search_results = index.query(
            namespace=req.url,
            vector=query_embedding,
            top_k=1,
            include_metadata=True
        )
        
        context = ""
        if search_results['matches']:
            context = search_results['matches'][0]['metadata']['text']
        
        prompt = f"You are a helpful assistant for the website {req.url}. Answer based ONLY on this context: {context}. User Question: {req.message}"
        
        response = model.generate_content(prompt)
        return {"reply": response.text}
        
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}
