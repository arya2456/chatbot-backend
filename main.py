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

# --- CONFIGURATION ---
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

class AutoSyncRequest(BaseModel):
    url: str
    client_id: str
    gemini_api_key: str

# --- HELPER: AUTO-DETECT MODEL ---
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

# --- CRAWLER LOGIC ---
async def fetch_url(session, url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
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
    """
    Crawls website and updates the Last Sync Timestamp in Pinecone
    """
    print(f"--- STARTING BACKGROUND SYNC FOR {client_id} ---")
    
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
                for script in soup(["script", "style", "nav", "footer", "iframe"]): 
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                
                if len(text) > 200:
                    scraped_data.append({"url": current_url, "text": text})

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    if is_internal_link(base_domain, full_url):
                        full_url = full_url.split('#')[0].rstrip('/')
                        if full_url not in visited and full_url not in to_visit:
                            if not any(x in full_url for x in ['.jpg', '.png', 'login', 'wp-admin', 'mailto', 'tel:']):
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

        # Save Vectors
        if vectors:
            index.upsert(vectors=vectors, namespace=client_id)
            
            # --- CRITICAL: SAVE THE SYNC TIMESTAMP ---
            # We save a dummy vector named "config_SYNC" to remember the time
            current_time = int(time.time())
            index.upsert(
                vectors=[{
                    "id": "config_SYNC",
                    "values": [0.1] * 768, # Dummy values
                    "metadata": {"last_sync_timestamp": current_time, "info": "DO NOT DELETE"}
                }],
                namespace=client_id
            )
            print(f"--- SYNC COMPLETE for {client_id} at {current_time} ---")
            return True

    except Exception as e:
        print(f"Sync Error: {e}")
        return False

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Chatbot Brain is Active", "mode": "Traffic-Triggered Sync"}

@app.post("/train")
async def train_bot(request: TrainRequest):
    success = await crawl_and_index(request.url, request.client_id, request.gemini_api_key)
    if success: return {"status": "success"}
    else: return {"status": "error"}

# --- NEW: TRAFFIC TRIGGERED SYNC ---
@app.post("/trigger-sync")
async def trigger_sync(request: AutoSyncRequest, background_tasks: BackgroundTasks):
    """
    Called by the widget.js when a user visits the site.
    Checks if 24 hours have passed since last sync.
    """
    try:
        # 1. Check "Memory" for last sync time
        fetch_response = index.fetch(ids=["config_SYNC"], namespace=request.client_id)
        
        should_sync = False
        current_time = int(time.time())
        
        if "config_SYNC" in fetch_response.vectors:
            last_sync = int(fetch_response.vectors["config_SYNC"].metadata.get("last_sync_timestamp", 0))
            # 86400 seconds = 24 Hours
            if (current_time - last_sync) > 86400:
                should_sync = True
                print(f"Time to sync {request.client_id}! Last sync was {last_sync}")
            else:
                print(f"Skipping sync for {request.client_id}. Up to date.")
        else:
            # Never synced before (or first time using this new system)
            should_sync = True
        
        if should_sync:
            background_tasks.add_task(crawl_and_index, request.url, request.client_id, request.gemini_api_key)
            return {"status": "Sync Started (Background)"}
            
        return {"status": "Up to date"}

    except Exception as e:
        return {"status": "error", "detail": str(e)}


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
            top_k=4, 
            include_metadata=True
        )

        context_parts = []
        for m in search_results['matches']:
            url = m['metadata'].get('url', '')
            text = m['metadata']['text']
            context_parts.append(f"URL: {url}\nINFO: {text}")
        
        context_str = "\n\n".join(context_parts)
        
        system_instruction = f"""
        You are a smart, efficient assistant for {request.client_id}.
        STRICT GUIDELINES:
        1. BE CONCISE: Maximum 2-3 sentences.
        2. BE DATA-DRIVEN: Use the Source URL provided.
        3. LINKS: Return clickable links (Markdown) if found.
        4. FALLBACK: If unsure, ask user to check website.
        
        CONTEXT:
        {context_str}
        """

        try:
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content(f"{system_instruction}\n\nUSER QUESTION: {request.message}")
            return {"answer": response.text}
        except:
            model = genai.GenerativeModel(get_best_model())
            response = model.generate_content(f"{system_instruction}\n\nUSER QUESTION: {request.message}")
            return {"answer": response.text}
        
    except Exception as e:
        return {"answer": f"System Error: {str(e)}"}
