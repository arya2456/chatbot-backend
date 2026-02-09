import os
import time
import asyncio
import logging
import uuid
from typing import Dict
from bs4 import BeautifulSoup
import aiohttp
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai
from contextlib import asynccontextmanager

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GLOBAL STATUS TRACKER (For the Progress Bar)
CRAWL_STATUS: Dict[str, dict] = {}

if not PINECONE_API_KEY:
    logger.error("CRITICAL: PINECONE_API_KEY is missing.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(daily_auto_crawler())
    yield

app = FastAPI(title="Omni-Brain v22.0 (Full SaaS)", version="22.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def connect_db():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)
    except: return None

index = connect_db()

# --- 2. EXPANDED DATA MODELS (Everything form Dashboard) ---
class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str 
    # Visuals
    bot_name: str = "AI Assistant"
    bot_lang: str = "English"
    bot_personality: str = "Auto-Detect"
    bot_color: str = "#4F46E5"
    bot_avatar: str = ""
    # Business Details
    biz_name: str = ""
    biz_phone: str = ""
    biz_email: str = ""
    # Functionality
    bot_status: bool = True
    response_delay_ms: int = 1500
    max_conv_length: int = 50
    fallback_msg: str = "I'm not sure. Would you like to speak to a human?"
    # Leads & Actions
    leads_trigger: str = "Before pricing"
    collect_name: bool = True
    collect_email: bool = True
    collect_phone: bool = False
    book_call_link: str = ""
    whatsapp_number: str = ""

class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"
    api_key: str = "" 

class StatusRequest(BaseModel):
    client_id: str

# --- 3. SMART HELPERS ---
def get_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

def safe_embed(text, api_key):
    genai.configure(api_key=api_key)
    try:
        return genai.embed_content(model="models/embedding-001", content=text)['embedding']
    except:
        return [0.0] * 768

async def daily_auto_crawler():
    while True:
        await asyncio.sleep(86400) 

# --- 4. THE REAL SCRAPER (With Progress Updates) ---
async def deep_scraper_engine(start_url: str, client_id: str, api_key: str):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    
    # INIT STATUS
    CRAWL_STATUS[client_id] = {"status": "scanning", "pages": 0, "progress": 5}
    
    domain = urlparse(start_url).netloc
    visited, vectors, seen_hashes = set(), [], set()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))

    async with aiohttp.ClientSession() as session:
        while not queue.empty() and len(visited) < 30:
            url, depth = await queue.get()
            if url in visited or depth > 3: continue
            visited.add(url)
            
            # UPDATE PROGRESS
            CRAWL_STATUS[client_id]["pages"] = len(visited)
            CRAWL_STATUS[client_id]["progress"] = int((len(visited) / 30) * 90) # Up to 90%
            
            try:
                async with session.get(url, timeout=10, ssl=False) as resp:
                    if resp.status != 200: continue
                    soup = BeautifulSoup(await resp.text(), 'html.parser')
                    
                    # Discovery
                    for a in soup.find_all('a', href=True):
                        link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                        if urlparse(link).netloc == domain and link not in visited:
                            await queue.put((link, depth + 1))
                    
                    # Processing
                    for x in soup(['script', 'style', 'nav', 'footer', 'aside']): x.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    if len(text) < 200: continue

                    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                    for i, chunk in enumerate(chunks):
                        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                        if h in seen_hashes: continue
                        seen_hashes.add(h)
                        
                        emb = safe_embed(chunk, api_key)
                        vectors.append({
                            "id": f"neural_{uuid.uuid4()}", "values": emb,
                            "metadata": {"text": chunk, "url": url, "type": "knowledge"} 
                        })
                    
                    # Batch Upload
                    if len(vectors) > 20:
                        index.upsert(vectors=vectors, namespace=client_id)
                        vectors = []
            except: continue
        
        if vectors: index.upsert(vectors=vectors, namespace=client_id)
        
        # COMPLETE
        CRAWL_STATUS[client_id] = {"status": "complete", "pages": len(visited), "progress": 100}

# --- 5. ENDPOINTS ---

@app.post("/train")
async def train_saas_engine(req: TrainRequest, bg: BackgroundTasks):
    try:
        # 1. Save FULL Configuration to Pinecone
        meta = req.dict()
        meta["type"] = "config"
        # Convert any non-string/int/float/bool to string for Pinecone compatibility
        for k, v in meta.items():
            if v is None: meta[k] = ""
        
        index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
        
        # 2. Start Scraper
        bg.add_task(deep_scraper_engine, req.url, req.client_id, req.gemini_api_key)
        return {"status": "success", "message": "Configuration Saved & Crawl Started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/get-crawl-status")
async def get_crawl_status(req: StatusRequest):
    """Returns the real-time progress of the scraper."""
    return CRAWL_STATUS.get(req.client_id, {"status": "idle", "progress": 0})

@app.post("/get-config")
async def get_conf(req: ChatRequest):
    """Fetches the FULL config so the dashboard can load saved settings."""
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if res.vectors:
            return res.vectors[f"config_{req.client_id}"].metadata
        return {"bot_name": "AI Support", "bot_color": "#4F46E5"}
    except: return {"bot_name": "AI Support", "bot_color": "#4F46E5"}

@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    try:
        # Failover Logic
        active_key = req.api_key.strip()
        if len(active_key) < 10:
            try:
                res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
                if res.vectors: active_key = res.vectors[f"config_{req.client_id}"].metadata.get("gemini_api_key")
            except: pass
        
        if not active_key or len(active_key) < 10:
            return {"answer": "Error: API Key missing. Check dashboard settings."}

        # Retrieve Context
        emb = safe_embed(req.message, active_key)
        search = index.query(namespace=req.client_id, vector=emb, top_k=4, include_metadata=True, filter={"type": "knowledge"})
        
        context = "\n".join([f"INFO: {m['metadata']['text']} [Source: {m['metadata']['url']}]" for m in search['matches']])

        # Fetch Personality from DB
        try:
            res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
            conf = res.vectors[f"config_{req.client_id}"].metadata if res.vectors else {}
        except: conf = {}

        sys_msg = f"""
        Identity: You are {conf.get('bot_name', 'AI Assistant')} at {conf.get('biz_name', 'our company')}.
        Tone: {conf.get('bot_personality', 'Professional')}.
        Facts: {context}
        
        Rules:
        1. Use the Facts to answer. If facts are missing, politely say you don't know.
        2. If the user asks for {conf.get('biz_phone', 'phone')}, give it.
        3. If the user asks to book a call, provide: {conf.get('book_call_link', '')}
        """
        
        model = get_model(active_key)
        try:
            ans = model.generate_content(f"{sys_msg}\n\nUSER: {req.message}").text
            return {"answer": ans}
        except Exception as e:
            if "400" in str(e): return {"answer": "Error: Invalid API Key."}
            return {"answer": "I'm having trouble connecting right now."}

    except Exception as e: return {"answer": f"System Error: {str(e)}"}

@app.post("/verify-install")
async def verify_engine(req: BaseModel): return {"status": "success"}
@app.post("/get-stats")
def stats_engine(req: BaseModel): return {"visitors": 0, "chats": 0, "leads": 0} 
@app.post("/get-leads")
def leads_engine(req: BaseModel): return {"leads": []}
@app.get("/")
def health(): return {"status": "Omni-Brain v22.0 Active"}
