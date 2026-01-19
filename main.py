# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from pinecone import Pinecone

app = FastAPI()

# Allow dashboard.fcmedia.in to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, change * to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Pinecone (Your Database)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# Ensure you created an index named 'chatbot-index' in Pinecone console
index = pc.Index(host=os.environ.get("PINECONE_HOST"))

class TrainRequest(BaseModel):
    url: str
    gemini_key: str

class ChatRequest(BaseModel):
    message: str
    url: str # We use the URL as the Client ID
    gemini_key: str

def get_clean_text(url):
    try:
        # Fake a browser visit so we don't get blocked
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Simple Logic to find brand color (looks for the first button color)
        # This is a basic implementation
        brand_color = "#000000" 
        button = soup.find('button')
        if button and button.get('style'):
            # This is a placeholder. A real CSS parser is more complex.
            pass 
            
        # Kill script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text()
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text[:10000] # Limit to 10k characters for the demo
    except Exception as e:
        return str(e)

@app.get("/")
def home():
    return {"status": "Chatbot Brain is Active"}

@app.post("/train")
def train_bot(req: TrainRequest):
    # 1. Scrape the site
    website_content = get_clean_text(req.url)
    
    # 2. Setup Gemini
    genai.configure(api_key=req.gemini_key)
    
    # 3. Create Embedding (Turn text into numbers)
    # We use 'embedding-001' model
    result = genai.embed_content(
        model="models/embedding-001",
        content=website_content,
        task_type="retrieval_document",
        title="Website Context"
    )
    
    # 4. Store in Pinecone
    # We use the URL as the namespace so clients don't mix data
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

@app.post("/chat")
def chat_bot(req: ChatRequest):
    try:
        # 1. Setup Gemini
        genai.configure(api_key=req.gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 2. Search Pinecone for context
        query_embedding = genai.embed_content(
            model="models/embedding-001",
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
        
        # 3. Ask Gemini
        prompt = f"You are a helpful assistant for the website {req.url}. Use this context to answer: {context}. User Question: {req.message}"
        
        response = model.generate_content(prompt)
        return {"reply": response.text}
        
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}
