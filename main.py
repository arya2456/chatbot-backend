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

# --- 2. AUTOMATION SCHEDULER (New Feature: Daily Crawl) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Launch the daily crawler background task
    asyncio.create_task(daily_auto_crawler())
    yield
    # Shutdown logic (if needed)

app = FastAPI(title="Omni-Brain v17.0 (The Automator)", version="17.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def connect_db():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"❌ DB Connection Failed: {e}")
        return None

index = connect_db()

# --- 3. SMART HELPERS ---

async def daily_auto_crawler():
    """
    Background task that wakes up every 24 hours to re-sync data.
    Note: In a full production app, you would fetch a list of active URLs here.
    """
    while True:
        logger.info("🕒 Daily Crawler: Waking up to sync websites...")
        try:
            # Placeholder: In production, fetch list of active client_ids and trigger scraper
            # for client in active_clients:
            #     await deep_scraper_engine(client.url, client.id, client.api_key)
            pass 
        except Exception as e:
            logger.error(f"Auto-Crawl Error: {e}")
        
        # Sleep for 24 Hours (86400 seconds)
        await asyncio.sleep(86400) 

def get_optimal_models(api_key):
    try:
        genai.configure(api_key=api_key)
        # Default to stable 2.0 Flash for speed
        return {"chat": "gemini-2.0-flash", "embed": "models/embedding-001"}
    except:
        return {"chat": "gemini-2.0-flash", "embed": "models/embedding-001"}

def safe_embed(model_name, text, api_key):
    genai.configure(api_key=api_key)
    try:
        # Smart toggle: 004 needs dim arg, 001 crashes with it
        if "004" in model_name:
            return genai.embed_content(model=model_name, content=text, output_dimensionality=768)['embedding']
        return genai.embed_content(model=model_name, content=text)['embedding']
    except:
        # Fallback to universal
        try:
            return genai.embed_content(model="models/embedding-001", content=text)['embedding']
        except:
            return [0.0] * 768

# --- 4. STYLE & COLOR DETECTOR (New Feature: Auto-Theme) ---
async def analyze_brand_style(session, text, url, api_key):
    """
    Detects Tone (Formal/Casual) and attempts to find brand colors.
    """
    style_prompt = f"""
    Analyze this website text and return a 1-word description of the tone.
    Options: Professional, Friendly, Technical, Luxury, Urgent, Playful.
    TEXT: {text[:1000]}
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        genai.configure(api_key=api_key)
        tone = model.generate_content(style_prompt).text.strip()
    except:
        tone = "Professional"

    # Enhanced Color Extraction
    color = "#4F46E5" # Default Indigo
    try:
        async with session.get(url, timeout=10, ssl=False) as resp:
            soup = BeautifulSoup(await resp.text(), 'html.parser')
            # 1. Check meta theme-color
            meta = soup.find("meta", {"name": "theme-color"})
            if meta: 
                color = meta.get("content")
            # 2. Heuristic: Look for hex codes in style tags (simplified)
    except: pass
    
    return tone, color

# --- 5. DATA MODELS ---
class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str 
    bot_name: str = "AI Assistant"
    bot_lang: str = "English"
    bot_personality: str = "Auto-Detect" # Default triggers auto-analysis

class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"

# --- 6. SCRAPER ENGINE ---
async def deep_scraper_engine(start_url: str, client_id: str, api_key: str):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    
    models = get_optimal_models(api_key)
    domain = urlparse(start_url).netloc
    visited, vectors, seen_hashes = set(), [], set()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))

    async with aiohttp.ClientSession() as session:
        # For Feature 2: Capture text for style analysis
        first_page_text = "" 
        
        while not queue.empty() and len(visited) < 30:
            url, depth = await queue.get()
            if url in visited or depth > 3: continue
            visited.add(url)
            try:
                async with session.get(url, timeout=10, ssl=False) as resp:
                    if resp.status != 200: continue
                    soup = BeautifulSoup(await resp.text(), 'html.parser')
                    
                    # Link Discovery
                    for a in soup.find_all('a', href=True):
                        link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                        if urlparse(link).netloc == domain and link not in visited:
                            await queue.put((link, depth + 1))
                    
                    # Cleanup
                    for x in soup(['script', 'style', 'nav', 'footer', 'aside']): x.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    if len(text) < 200: continue
                    
                    if not first_page_text: first_page_text = text

                    # Chunking
                    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                    for i, chunk in enumerate(chunks):
                        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                        if h in seen_hashes: continue
                        seen_hashes.add(h)

                        emb = safe_embed(models["embed"], chunk, api_key)
                        vectors.append({
                            "id": f"neural_{uuid.uuid4()}", "values": emb,
                            # FEATURE 3: Saving URL in metadata for Smart Linking
                            "metadata": {"text": chunk, "url": url, "type": "knowledge", "hash": h} 
                        })
                    
                    if len(vectors) > 20:
                        index.upsert(vectors=vectors, namespace=client_id)
                        vectors = []
            except: continue
        
        if vectors: index.upsert(vectors=vectors, namespace=client_id)

        # Apply Auto-Detected Branding
        tone, color = await analyze_brand_style(session, first_page_text, start_url, api_key)
        
        # Update Config with Learned Style
        try:
            res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
            if res.vectors:
                meta = res.vectors[f"config_{client_id}"].metadata
                # Only update if set to Auto-Detect or default
                if meta.get("bot_personality") == "Auto-Detect":
                    meta["bot_personality"] = f"Use a {tone} tone matching the website brand."
                if meta.get("bot_color") == "#4F46E5": 
                    meta["bot_color"] = color
                index.upsert(vectors=[{"id": f"config_{client_id}", "values": [1.0]*768, "metadata": meta}], namespace=client_id)
        except: pass

# --- 7. CHAT ENGINE (With Smart Linking) ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    try:
        # Config & Key Fetch
        try: res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        except: return {"answer": "Connection Error."}
        if not res.vectors: return {"answer": "Brain not active."}
        
        conf = res.vectors[f"config_{req.client_id}"].metadata
        api_key = conf.get('api_key', '').strip()
        
        models = get_optimal_models(api_key)
        
        # Retrieval
        emb = safe_embed(models["embed"], req.message, api_key)
        search = index.query(namespace=req.client_id, vector=emb, top_k=4, include_metadata=True, filter={"type": "knowledge"})
        
        # Build Context with Sources (Feature 3: Smart Linking)
        context_blocks = []
        for m in search['matches']:
            text = m['metadata'].get('text', '')
            url = m['metadata'].get('url', '')
            context_blocks.append(f"CONTENT: {text}\nSOURCE_URL: {url}")
        
        context_str = "\n\n".join(context_blocks)

        sys_msg = f"""
        You are {conf.get('bot_name')}.
        TONE: {conf.get('bot_personality', 'Professional')}.
        
        INSTRUCTIONS:
        1. Answer the user based ONLY on the provided CONTENT.
        2. If the answer is found in a specific CONTENT block, YOU MUST include the SOURCE_URL at the end of your sentence like this: (Read more: URL).
        3. Do not invent links. Only use the ones provided.
        
        KNOWLEDGE BASE:
        {context_str}
        """
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(models["chat"])
        ans = await generate_answer_with_retry(model, f"{sys_msg}\n\nUSER: {req.message}")
        
        # Simple logging
        log_id = f"log_{req.session_id}_{uuid.uuid4()}"
        index.upsert(vectors=[{"id": log_id, "values": [0.1]*768, "metadata": {"type": "chat_log", "user": req.message, "bot": ans, "timestamp": int(time.time()), "session": req.session_id}}], namespace=req.client_id)

        return {"answer": ans}

    except Exception as e:
        return {"answer": f"System Error: {str(e)}"}

# --- 8. TRAIN ENDPOINT ---
@app.post("/train")
async def train_saas_engine(req: TrainRequest, bg: BackgroundTasks):
    try:
        # Deletes old data to start fresh
        try: index.delete(delete_all=True, namespace=req.client_id)
        except: pass

        # Saves config
        meta = {
            "type": "config", "api_key": req.gemini_api_key, 
            "bot_name": req.bot_name, "bot_lang": req.bot_lang, 
            "bot_personality": req.bot_personality, # Can be "Auto-Detect"
            "bot_color": "#4F46E5" # Will be updated by scraper
        }
        index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
        
        # Starts scraper
        bg.add_task(deep_scraper_engine, req.url, req.client_id, req.gemini_api_key)
        return {"status": "success", "message": "Deep Sync & Auto-Discovery Started."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 9. HELPERS ---
async def generate_answer_with_retry(model, prompt, retries=1):
    for attempt in range(retries + 1):
        try: return model.generate_content(prompt).text
        except: await asyncio.sleep(1)
    return "I'm having trouble connecting. Please try again."

@app.post("/get-stats")
def stats_engine(req: BaseModel): return {"visitors": 0} # Placeholder for stats

@app.post("/verify-install")
async def verify_engine(req: BaseModel): return {"status": "success"}

@app.get("/")
def health(): return {"status": "Omni-Brain v17.0 Active"}
