import os
import asyncio
import aiohttp
import traceback
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

# --- CONFIGURATION ---
# ENSURE THIS MATCHES YOUR PINECONE DASHBOARD EXACTLY
PINECONE_INDEX_NAME = "chatbot-index" 

# PASTE YOUR KEY INSIDE THE QUOTES BELOW
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE CONNECTION ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Server Start Error: {e}")

# --- DATA MODELS ---
class TrainRequest(BaseModel):
    url: str
    client_id: str 
    gemini_api_key: str

class ChatRequest(BaseModel):
    message: str
    client_id: str
    gemini_api_key: str

# --- POWERFUL CRAWLER (Now pretends to be a human) ---
async def fetch_url(session, url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        async with session.get(url, headers=headers, timeout=15) as response:
            if response.status != 200: return None, url
            return await response.text(), url
    except Exception as e:
        print(f"Failed to crawl {url}: {e}")
        return None, url

async def crawl_website(base_url: str, max_pages: int = 15):
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
                for script in soup(["script", "style", "nav", "footer"]): 
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                
                if len(text) > 100:
                    scraped_data.append({"url": current_url, "text": text})

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith(base_url) or href.startswith('/'):
                        full_url = href if href.startswith('http') else base_url.rstrip('/') + href
                        if full_url not in visited:
                            to_visit.add(full_url)
    return scraped_data

# --- API ENDPOINTS ---

@app.post("/train")
async def train_bot(request: TrainRequest):
    print(f"Starting training for {request.client_id}")
    
    # 1. Crawl
    try:
        pages = await crawl_website(request.url)
        if not pages:
            return {"status": "error", "detail": "Could not read website. Ensure URL is correct (http/https)."}
    except Exception as e:
        return {"status": "error", "detail": f"Crawling crashed: {str(e)}"}

    # 2. Embed and Store
    try:
        genai.configure(api_key=request.gemini_api_key)
        vectors = []
        
        for page in pages:
            text = page['text'][:4000] # Increased chunk size for more "Power"
            
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )
            embedding = result['embedding']
            vector_id = f"{request.client_id}_{abs(hash(page['url']))}"
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {"text": text, "url": page['url']}
            })

        if vectors:
            index.upsert(vectors=vectors, namespace=request.client_id)
            
    except Exception as e:
         # This will now tell you EXACTLY what is wrong in the browser
         return {"status": "error", "detail": f"AI/Database Error: {str(e)}"}

    return {"status": "success", "pages": len(pages)}

@app.post("/chat")
async def chat_bot(request: ChatRequest):
    try:
        genai.configure(api_key=request.gemini_api_key)
        
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=request.message,
            task_type="retrieval_query",
        )['embedding']

        search_results = index.query(
            namespace=request.client_id,
            vector=embedding,
            top_k=5, # Increased context for smarter answers
            include_metadata=True
        )

        context = "\n".join([m['metadata']['text'] for m in search_results['matches']])
        
        # Using the Smartest Model: gemini-1.5-flash
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Context from website:\n{context}\n\nUser Question: {request.message}\nAnswer as a helpful support agent:"
        
        response = model.generate_content(prompt)
        return {"answer": response.text}
        
    except Exception as e:
        # Returns the actual error message so we can fix it
        return {"answer": f"System Error: {str(e)}"}
