import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

# --- CONFIGURATION ---
# PASTE YOUR PINECONE KEY HERE IF NOT SET IN RENDER ENVIRONMENT
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W") 
PINECONE_INDEX_NAME = "chatbot-index"

app = FastAPI()

# Allow your cPanel and Client sites to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE SETUP ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Pinecone Error: {e}")

# --- DATA MODELS ---
# This is what matches your new Dashboard!
class TrainRequest(BaseModel):
    url: str
    client_id: str 
    gemini_api_key: str

class ChatRequest(BaseModel):
    message: str
    client_id: str
    gemini_api_key: str

# --- CRAWLER ENGINE ---
async def fetch_url(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            return await response.text(), url
    except:
        return None, url

async def crawl_website(base_url: str, max_pages: int = 10):
    visited = set()
    to_visit = {base_url}
    scraped_data = []

    async with aiohttp.ClientSession() as session:
        while to_visit and len(visited) < max_pages:
            batch = list(to_visit)[:5] 
            for u in batch: to_visit.remove(u)
            
            tasks = [fetch_url(session, url) for url in batch]
            results = await asyncio.gather(*tasks)

            for html, current_url in results:
                if not html: continue
                visited.add(current_url)
                
                soup = BeautifulSoup(html, 'html.parser')
                
                # Cleanup HTML
                for script in soup(["script", "style", "nav", "footer"]): 
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                
                if len(text) > 100:
                    scraped_data.append({"url": current_url, "text": text})

                # Find links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith(base_url) or href.startswith('/'):
                        full_url = href if href.startswith('http') else base_url.rstrip('/') + href
                        if full_url not in visited:
                            to_visit.add(full_url)
    
    return scraped_data

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Chatbot Brain is Active"}

@app.post("/train")
async def train_bot(request: TrainRequest):
    print(f"Training for: {request.client_id} on {request.url}")
    
    # 1. Crawl
    try:
        pages = await crawl_website(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Crawling failed: {str(e)}")
    
    if not pages:
        raise HTTPException(status_code=400, detail="Could not read website. Check URL.")

    # 2. Embed and Store
    try:
        genai.configure(api_key=request.gemini_api_key)
        vectors = []
        
        for page in pages:
            text = page['text'][:2000] # Limit chunk size
            
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )
            embedding = result['embedding']
            
            # Simple ID generation
            vector_id = f"{request.client_id}_{abs(hash(page['url']))}"
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {"text": text, "url": page['url']}
            })

        # Upsert to Pinecone with Namespace
        if vectors:
            index.upsert(vectors=vectors, namespace=request.client_id)
            
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

    return {"status": "success", "pages": len(pages)}

@app.post("/chat")
async def chat_bot(request: ChatRequest):
    try:
        genai.configure(api_key=request.gemini_api_key)
        
        # 1. Embed Question
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=request.message,
            task_type="retrieval_query",
        )['embedding']

        # 2. Search Pinecone
        search_results = index.query(
            namespace=request.client_id,
            vector=embedding,
            top_k=3,
            include_metadata=True
        )

        context = "\n".join([m['metadata']['text'] for m in search_results['matches']])

        # 3. Generate Answer
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        prompt = f"Context: {context}\n\nUser Question: {request.message}\nAnswer:"
        
        response = model.generate_content(prompt)
        return {"answer": response.text}
        
    except Exception as e:
        return {"answer": "I'm having trouble thinking right now."}
