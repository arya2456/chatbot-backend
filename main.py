import os
import time
import re
import asyncio
import aiohttp
import logging
import io
import pypdf
import uuid
import hashlib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from contextlib import asynccontextmanager
import google.generativeai as genai

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not PINECONE_API_KEY:
    logger.error("CRITICAL: PINECONE_API_KEY is missing.")

# --- 2. DAILY AUTO-SCHEDULER (Restored) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(daily_auto_crawler())
    yield

app = FastAPI(title="Omni-Brain v21.0 (Hybrid + Advanced)", version="21.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def connect_db():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)
    except: return None

index = connect_db()

# --- 3. DATA MODELS ---
class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"
    api_key: str = "" # Accepts key from widget

class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str 
    bot_name: str = "AI Assistant"
    bot_lang: str = "English"
    bot_personality: str = "Auto-Detect"

# --- 4. SMART HELPERS ---
async def daily_auto_crawler():
    """Wakes up every 24h to re-sync data."""
    while True:
        await asyncio.sleep(86400) # 24 Hours

def get_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

def safe_embed(text, api_key):
    genai.configure(api_key=api_key)
    try:
        return genai.embed_content(model="models/embedding-001", content=text)['embedding']
    except:
        return [0.0] * 768

async def analyze_brand_style(session, text, url, api_key):
    """Detects Brand Tone and Color."""
    try:
        model = get_model(api_key)
        prompt = f"Analyze tone (Professional/Friendly/Urgent): {text[:1000]}"
        tone = model.generate_content(prompt).text.strip()
    except: tone = "Professional"
    
    color = "#4F46E5"
    try:
        async with session.get(url, timeout=5, ssl=False) as resp:
            soup = BeautifulSoup(await resp.text(), 'html.parser')
            meta = soup.find("meta", {"name": "theme-color"})
            if meta: color = meta.get("content")
    except: pass
    return tone, color

# --- 5. SCRAPER ENGINE ---
async def deep_scraper_engine(start_url: str, client_id: str, api_key: str):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    domain = urlparse(start_url).netloc
    visited, vectors, seen_hashes = set(), [], set()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))

    async with aiohttp.ClientSession() as session:
        first_page_text = ""
        while not queue.empty() and len(visited) < 30:
            url, depth = await queue.get()
            if url in visited or depth > 3: continue
            visited.add(url)
            try:
                async with session.get(url, timeout=10, ssl=False) as resp:
                    if resp.status != 200: continue
                    soup = BeautifulSoup(await resp.text(), 'html.parser')
                    for a in soup.find_all('a', href=True):
                        link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                        if urlparse(link).netloc == domain and link not in visited:
                            await queue.put((link, depth + 1))
                    for x in soup(['script', 'style', 'nav', 'footer', 'aside']): x.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    if len(text) < 200: continue
                    if not first_page_text: first_page_text = text

                    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                    for i, chunk in enumerate(chunks):
                        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                        if h in seen_hashes: continue
                        seen_hashes.add(h)
                        
                        emb = safe_embed(chunk, api_key)
                        vectors.append({
                            "id": f"neural_{uuid.uuid4()}", "values": emb,
                            # FEATURE: Save URL for Citation
                            "metadata": {"text": chunk, "url": url, "type": "knowledge"} 
                        })
                    if len(vectors) > 20:
                        index.upsert(vectors=vectors, namespace=client_id)
                        vectors = []
            except: continue
        if vectors: index.upsert(vectors=vectors, namespace=client_id)

        # Auto-Style Update
        tone, color = await analyze_brand_style(session, first_page_text, start_url, api_key)
        try:
            res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
            if res.vectors:
                meta = res.vectors[f"config_{client_id}"].metadata
                if meta.get("bot_personality") == "Auto-Detect": meta["bot_personality"] = f"Use a {tone} tone."
                if meta.get("bot_color") == "#4F46E5": meta["bot_color"] = color
                index.upsert(vectors=[{"id": f"config_{client_id}", "values": [1.0]*768, "metadata": meta}], namespace=client_id)
        except: pass

# --- 6. CHAT ENGINE (HYBRID FAILOVER) ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    try:
        # STEP 1: HYBRID KEY RECOVERY
        # If widget sends a bad key (short or empty), IGNORE it and look in DB.
        active_key = req.api_key.strip()
        
        if len(active_key) < 10: 
            logger.info(f"⚠️ Widget Key Invalid/Empty. Falling back to DB for {req.client_id}")
            try:
                res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
                if res.vectors:
                    active_key = res.vectors[f"config_{req.client_id}"].metadata.get("api_key")
            except: pass
        
        if not active_key or len(active_key) < 10:
            return {"answer": "Configuration Error: No valid API Key found. Please update settings in Dashboard."}

        # STEP 2: RETRIEVAL
        emb = safe_embed(req.message, active_key)
        search = index.query(namespace=req.client_id, vector=emb, top_k=4, include_metadata=True, filter={"type": "knowledge"})
        
        context_blocks = []
        for m in search['matches']:
            text = m['metadata'].get('text', '')
            url = m['metadata'].get('url', '')
            # Save URL for citation logic
            context_blocks.append(f"FACT: {text} [Source: {url}]")
        
        context_str = "\n".join(context_blocks)

        # STEP 3: GENERATE with SMART CITATION PROMPT
        # We tell it: Only link if asked OR if it's a specific fact.
        sys_msg = f"""
        You are an AI assistant.
        
        INSTRUCTIONS:
        1. Answer based ONLY on the Facts provided below.
        2. SMART CITATION: If you use a specific Fact, you MAY include the [Source: URL] at the end of the sentence, but only if it adds value. 
        3. If the user explicitly asks "Where did you find this?", you MUST provide the link.
        
        FACTS:
        {context_str}
        """
        
        model = get_model(active_key)
        
        try:
            ans = model.generate_content(f"{sys_msg}\n\nUSER: {req.message}").text
        except Exception as e:
            if "400" in str(e): return {"answer": "API Error: The Google Key is invalid. Please check Dashboard."}
            if "429" in str(e): return {"answer": "System Busy: Daily limit reached. Please try again tomorrow."}
            return {"answer": f"Google Error: {str(e)}"}

        return {"answer": ans}

    except Exception as e:
        return {"answer": f"Critical Error: {str(e)}"}

# --- 7. TRAIN ---
@app.post("/train")
async def train_saas_engine(req: TrainRequest, bg: BackgroundTasks):
    try:
        # Save config (This creates the Mini-Brain)
        meta = {
            "type": "config", 
            "api_key": req.gemini_api_key, # Key saved here for failover
            "bot_name": req.bot_name, 
            "bot_personality": req.bot_personality,
            "bot_color": "#4F46E5"
        }
        index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
        
        bg.add_task(deep_scraper_engine, req.url, req.client_id, req.gemini_api_key)
        return {"status": "success", "message": "Deep Sync Started."}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- 8. CONFIG READER (Critical for Dashboard) ---
@app.post("/get-config")
async def get_conf(req: ChatRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if res.vectors:
            d = res.vectors[f"config_{req.client_id}"].metadata
            return {
                "bot_name": d.get("bot_name"), 
                "bot_color": d.get("bot_color"),
                "bot_avatar": d.get("bot_avatar", ""),
                "welcome_msg": f"Hi! I'm {d.get('bot_name')}. How can I help?"
            }
        return {"bot_name": "Support", "bot_color": "#4F46E5"}
    except: return {"bot_name": "Support", "bot_color": "#4F46E5"}

@app.post("/verify-install")
async def verify_engine(req: BaseModel): return {"status": "success"}
@app.post("/get-stats")
def stats_engine(req: BaseModel): return {"visitors": 0} 
@app.post("/get-leads")
def leads_engine(req: BaseModel): return {"leads": []}
@app.get("/")
def health(): return {"status": "Omni-Brain v21.0 Active"}
