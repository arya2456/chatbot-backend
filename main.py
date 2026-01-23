import os
import asyncio
import aiohttp
import time
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

# --- CONFIG ---
PINECONE_INDEX_NAME = "chatbot-index" 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB SETUP ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Server Start Error: {e}")

# --- MODELS ---
class TrainRequest(BaseModel):
    url: str
    client_id: str 
    gemini_api_key: str

class ChatRequest(BaseModel):
    message: str
    client_id: str

class AutoSyncRequest(BaseModel):
    url: str
    client_id: str

# --- HELPERS ---
def save_client_key(client_id, api_key):
    try:
        # FIX: Changed [0.0] to [1.0] to satisfy Pinecone requirements
        index.upsert(
            vectors=[{
                "id": f"config_{client_id}", 
                "values": [1.0] * 768, 
                "metadata": {"api_key": api_key, "type": "config"}
            }],
            namespace=client_id
        )
    except Exception as e:
        print(f"Error saving key: {e}")

def get_client_key(client_id):
    try:
        response = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if f"config_{client_id}" in response.vectors:
            return response.vectors[f"config_{client_id}"].metadata.get("api_key")
        return None
    except:
        return None

def get_best_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name: return m.name
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                return m.name
    except:
        return "models/gemini-1.5-flash"
    return "models/gemini-pro"

# --- CRAWLER ---
async def fetch_url(session, url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            if response.status != 200: return None, url
            return await response.text(), url
    except:
        return None, url

def is_internal_link(base_domain, link_url):
    link_parts = urlparse(link_url)
    link_domain = link_parts.netloc.replace("www.", "")
    base_clean = base_domain.replace("www.", "")
    return base_clean in link_domain or link_domain == ""

async def crawl_and_index(url: str, client_id: str, api_key: str):
    print(f"--- SYNC START: {client_id} ---")
    
    # SAVE KEY FIRST
    save_client_key(client_id, api_key)
    
    if not url.startswith('http'): url = 'https://' + url
    base_domain = urlparse(url).netloc
    visited = set()
    to_visit = {url}
    scraped_data = []

    async with aiohttp.ClientSession() as session:
        while to_visit and len(visited) < 50:
            batch = list(to_visit)[:8] 
            for u in batch: to_visit.remove(u)
            tasks = [fetch_url(session, u) for u in batch]
            results = await asyncio.gather(*tasks)

            for html, current_url in results:
                if not html: continue
                visited.add(current_url)
                soup = BeautifulSoup(html, 'html.parser')
                for script in soup(["script", "style", "nav", "footer", "iframe"]): script.extract()
                text = soup.get_text(separator=' ', strip=True)
                if len(text) > 200:
                    scraped_data.append({"url": current_url, "text": text})

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    if is_internal_link(base_domain, full_url):
                        full_url = full_url.split('#')[0].rstrip('/')
                        if full_url not in visited and full_url not in to_visit:
                            if not any(x in full_url for x in ['.jpg', '.png', 'login', 'wp-admin']):
                                to_visit.add(full_url)
    
    if not scraped_data: return False

    try:
        genai.configure(api_key=api_key)
        vectors = []
        for page in scraped_data:
            text = page['text'][:4000]
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )
            vector_id = f"{client_id}_{abs(hash(page['url']))}"
            vectors.append({
                "id": vector_id,
                "values": result['embedding'],
                "metadata": {"text": text, "url": page['url']}
            })

        if vectors:
            index.upsert(vectors=vectors, namespace=client_id)
            current_time = int(time.time())
            # FIX: Changed [0.0] to [1.0] here as well
            index.upsert(
                vectors=[{
                    "id": "config_SYNC",
                    "values": [1.0] * 768, 
                    "metadata": {"last_sync_timestamp": current_time}
                }],
                namespace=client_id
            )
            return True
    except Exception as e:
        print(f"Sync Error: {e}")
        return False

# --- API ---
@app.get("/")
def home(): return {"status": "Secure Brain Active"}

@app.post("/train")
async def train_bot(request: TrainRequest):
    success = await crawl_and_index(request.url, request.client_id, request.gemini_api_key)
    if success: return {"status": "success"}
    else: return {"status": "error"}

@app.post("/trigger-sync")
async def trigger_sync(request: AutoSyncRequest, background_tasks: BackgroundTasks):
    api_key = get_client_key(request.client_id)
    if not api_key: return {"status": "No Key Found"}
    
    background_tasks.add_task(crawl_and_index, request.url, request.client_id, api_key)
    return {"status": "Sync Started"}

@app.post("/chat")
async def chat_bot(request: ChatRequest):
    api_key = get_client_key(request.client_id)
    if not api_key: return {"answer": "Security Error: Please re-train bot."}

    try:
        genai.configure(api_key=api_key)
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=request.message,
            task_type="retrieval_query",
        )['embedding']

        search_results = index.query(
            namespace=request.client_id, vector=embedding, top_k=4, include_metadata=True
        )

        context = "\n\n".join([f"URL: {m['metadata'].get('url','')}\nINFO: {m['metadata']['text']}" for m in search_results['matches']])
        
        system = f"You are a helpful assistant for {request.client_id}. Use this context:\n{context}\n\nBe concise and use 'We'."
        
        try:
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            res = model.generate_content(f"{system}\n\nQuestion: {request.message}")
            return {"answer": res.text}
        except:
            model = genai.GenerativeModel(get_best_model())
            res = model.generate_content(f"{system}\n\nQuestion: {request.message}")
            return {"answer": res.text}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
