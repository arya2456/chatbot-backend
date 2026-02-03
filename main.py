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
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

# --- 1. SYSTEM CONFIGURATION ---
# Security: Load keys from Environment Variables (set these in Render)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")

# Optional: Backend Auth Key for securing your Dashboard <-> Brain connection
API_AUTH_KEY = os.getenv("BACKEND_API_KEY") 

if not PINECONE_API_KEY:
    raise RuntimeError("CRITICAL: PINECONE_API_KEY is missing in Environment Variables.")

# Basic Auth Dependency (Optional usage)
def verify_api_key(x_api_key: str = Header(None)):
    if API_AUTH_KEY and x_api_key != API_AUTH_KEY:
        raise HTTPException(status_code=401, detail="Invalid Backend API Key")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Omni-Brain v13.0 SaaS Enterprise", version="13.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- DATABASE CONNECTION ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    logger.info("✅ Pinecone DB Connected")
except Exception as e:
    logger.error(f"❌ Database Connection Failed: {e}")
    index = None

# --- DATA MODELS ---
class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str = "" # Optional: If empty, system tries to recall existing key
    bot_name: str = "AI Assistant"
    bot_lang: str = "English"
    timezone: str = "Auto-detect (IST)"
    bot_status: bool = True
    biz_name: str = ""
    biz_phone: str = ""
    biz_email: str = ""
    collect_name: bool = True
    collect_email: bool = True
    collect_phone: bool = True
    collect_company: bool = False
    trigger_strategy: str = "Before sharing pricing"
    book_call_active: bool = False
    book_call_link: str = ""
    whatsapp_active: bool = False
    whatsapp_number: str = ""
    max_conv_length: int = 50
    response_delay_ms: int = 1000
    fallback_msg: str = "I'm not sure about that. Would you like to speak to a human agent?"
    bot_personality: str = "Professional"
    bot_color: str = "#4F46E5"

class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"
    page_url: str = ""

class AutoSyncRequest(BaseModel):
    client_id: str
    url: str = ""

# --- HELPER: GENAI RETRY LOGIC ---
async def generate_answer_with_retry(model, prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            return model.generate_content(prompt).text
        except Exception as e:
            if attempt == retries: raise
            await asyncio.sleep(0.5 * (attempt + 1))

# --- HELPER: THEME EXTRACTION ---
async def extract_theme_color(session, url):
    """Scans host website for brand colors."""
    try:
        async with session.get(url, timeout=10, ssl=False) as resp:
            if resp.status != 200: return "#4F46E5"
            soup = BeautifulSoup(await resp.text(), 'html.parser')
            meta = soup.find("meta", {"name": "theme-color"})
            return meta.get("content") if meta else "#4F46E5"
    except: return "#4F46E5"

# --- HELPER: CONVERSATION MEMORY ---
def get_conversation_history(client_id: str, session_id: str, limit: int = 6):
    """Fetches previous chat context for Multi-turn conversation."""
    try:
        dummy = [0.0] * 768
        res = index.query(
            namespace=client_id, vector=dummy, top_k=limit, 
            filter={"type": "chat_log", "session": session_id}, 
            include_metadata=True
        )
        # Sort by timestamp ascending
        matches = sorted(res.get("matches", []), key=lambda m: m["metadata"].get("timestamp", 0))
        history = []
        for m in matches:
            history.append(f"User: {m['metadata'].get('user','')}\nAI: {m['metadata'].get('bot','')}")
        return "\n".join(history)
    except: return ""

# --- STEP 1: ROBUST SAAS CRAWLER ---
async def deep_scraper_engine(start_url: str, client_id: str, api_key: str, max_pages: int = 40):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    
    # Configure Client Key for this background task
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"Scraper Config Error: {e}")
        return

    domain = urlparse(start_url).netloc
    visited, vectors, seen_hashes = set(), [], set()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))

    async with aiohttp.ClientSession() as session:
        # Auto-Theme Update
        color = await extract_theme_color(session, start_url)
        try:
            res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
            if res.vectors:
                meta = res.vectors[f"config_{client_id}"].metadata
                if meta.get("bot_color") == "#4F46E5":
                    meta["bot_color"] = color
                    index.upsert(vectors=[{"id": f"config_{client_id}", "values": [1.0]*768, "metadata": meta}], namespace=client_id)
        except: pass

        while not queue.empty() and len(visited) < max_pages:
            url, depth = await queue.get()
            if url in visited or depth > 3: continue
            visited.add(url)
            try:
                async with session.get(url, timeout=15, ssl=False) as resp:
                    if resp.status != 200: continue
                    soup = BeautifulSoup(await resp.text(), 'html.parser')
                    
                    # 1. Harvest Links
                    for a in soup.find_all('a', href=True):
                        link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                        if urlparse(link).netloc == domain and link not in visited:
                            await queue.put((link, depth + 1))

                    # 2. Clean HTML
                    for x in soup(['script', 'style', 'nav', 'footer', 'iframe', 'aside']): x.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    if len(text) < 200: continue

                    # 3. AI Cleaning (Gemini)
                    cleaner = genai.GenerativeModel("gemini-2.5-flash")
                    clean_text = await generate_answer_with_retry(cleaner, f"Extract business facts only. Remove fluff/navigation. TEXT: {text[:8000]}")

                    # 4. Chunking & Deduplication
                    chunks = [clean_text[i:i+1500] for i in range(0, len(clean_text), 1500)]
                    for i, chunk in enumerate(chunks):
                        # Hash check
                        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                        if h in seen_hashes: continue
                        seen_hashes.add(h)

                        emb = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
                        vectors.append({
                            "id": f"neural_{uuid.uuid4()}", # Robust ID
                            "values": emb,
                            "metadata": {"text": chunk, "url": url, "type": "knowledge", "hash": h, "source": "recursive_crawl"}
                        })
                    
                    if len(vectors) > 30:
                        index.upsert(vectors=vectors, namespace=client_id)
                        vectors = []
            except: continue
    if vectors: index.upsert(vectors=vectors, namespace=client_id)

# --- DOCUMENT UPLOAD ---
@app.post("/upload-file")
async def upload_file_engine(client_id: str, file: UploadFile = File(...)):
    try:
        # Fetch Key
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if not res.vectors: return {"status": "error", "message": "Train website first."}
        
        api_key = res.vectors[f"config_{client_id}"].metadata.get('api_key')
        if not api_key: return {"status": "error", "message": "API Key missing."}
        
        genai.configure(api_key=api_key)

        content = ""
        if file.content_type == "application/pdf":
            reader = pypdf.PdfReader(io.BytesIO(await file.read()))
            for p in reader.pages: content += p.extract_text() + "\n"
        else: content = (await file.read()).decode("utf-8")

        chunks = [content[i:i+1200] for i in range(0, len(content), 1200)]
        vectors = []
        for i, c in enumerate(chunks):
             emb = genai.embed_content(model="models/text-embedding-004", content=c)['embedding']
             vectors.append({
                 "id": f"doc_{uuid.uuid4()}",
                 "values": emb,
                 "metadata": {"text": c, "source": file.filename, "type": "knowledge", "category": "uploaded_document"}
             })
        index.upsert(vectors=vectors, namespace=client_id)
        return {"status": "success", "message": f"Learned from {file.filename}"}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- CHAT ENGINE (MEMORY + SECURITY) ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    try:
        # 1. Fetch Config
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if not res.vectors: return {"answer": "Initializing... please refresh."}
        conf = res.vectors[f"config_{req.client_id}"].metadata
        
        # 2. Status Check
        if str(conf.get("bot_status", "True")).lower() in ("false", "0", "off"):
            return {"answer": "This assistant is currently offline."}

        # 3. Configure Key
        api_key = conf.get('api_key')
        if not api_key: return {"answer": "System Error: API Key missing. Please contact support."}
        genai.configure(api_key=api_key)

        await asyncio.sleep(int(conf.get("delay", 1000)) / 1000)

        # 4. Lead Gating (Email)
        if conf.get("leads_trigger") == "Before sharing pricing" and conf.get("collect_email", True):
             if any(x in req.message.lower() for x in ["price", "cost", "how much", "fees"]):
                 # Check if lead exists in session
                 dummy = [0.0]*768
                 ex = index.query(namespace=req.client_id, vector=dummy, top_k=1, filter={"type": "lead", "session": req.session_id})
                 if not ex.get("matches"):
                     return {"answer": "I'd be happy to share pricing! Could you please share your email address first?"}

        # 5. Fetch Context (Memory & RAG)
        history = get_conversation_history(req.client_id, req.session_id)
        
        emb = genai.embed_content(model="models/text-embedding-004", content=req.message)['embedding']
        search = index.query(namespace=req.client_id, vector=emb, top_k=6, include_metadata=True, filter={"type": "knowledge"})
        ctx = "\n\n".join([m['metadata']['text'] for m in search['matches']])

        # 6. Strict Prompt
        sys_msg = f"""
        IDENTITY: You are {conf.get('bot_name')} at "{conf.get('biz_name')}". 
        LANGUAGE: Answer strictly in {conf.get('bot_lang', 'English')}.
        RULES:
        - 1-3 lines max.
        - No 'Acme'. No hallucinations.
        - Use HISTORY to remember names/context.
        - If missing info: "{conf.get('fallback')}"
        
        KNOWLEDGE: {ctx}
        HISTORY: {history}
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        ans = await generate_answer_with_retry(model, f"{sys_msg}\n\nUSER: {req.message}")

        # 7. Secure Logging (UUID)
        log_id = f"log_{req.session_id}_{uuid.uuid4()}"
        m_type = "lead" if re.search(r"[\w\.-]+@[\w\.-]+", req.message) else "chat_log"
        
        meta = {
            "type": m_type, 
            "user": req.message, 
            "bot": ans, 
            "session": req.session_id, 
            "timestamp": int(time.time()),
            "email": req.message if m_type == "lead" else ""
        }
        
        index.upsert(vectors=[{"id": log_id, "values": [0.1]*768, "metadata": meta}], namespace=req.client_id)
        return {"answer": ans}
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return {"answer": "I'm optimizing my neural links. Try again?"}

# --- TRAIN ENGINE (KEY PRESERVATION + WIPE) ---
@app.post("/train")
async def train_saas_engine(req: TrainRequest, bg: BackgroundTasks):
    try:
        # 1. KEY LOGIC: "Handshake"
        # If dashboard sends key, use it. If not, fetch existing from Brain.
        final_api_key = req.gemini_api_key
        
        if not final_api_key:
            existing = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
            if existing.vectors:
                final_api_key = existing.vectors[f"config_{req.client_id}"].metadata.get("api_key")
        
        if not final_api_key:
            return {"status": "error", "message": "Critical: No API Key found. Admin must configure it first."}

        # 2. Wipe Old Memory (Safe because we have the key)
        try: index.delete(delete_all=True, namespace=req.client_id)
        except: pass

        # 3. Save Config (Persist Key)
        meta = {
            "type": "config", 
            "api_key": final_api_key, 
            "bot_name": req.bot_name, "bot_lang": req.bot_lang, 
            "bot_status": str(req.bot_status), "biz_name": req.biz_name, 
            "biz_phone": req.biz_phone, "biz_email": req.biz_email,
            "leads_trigger": req.trigger_strategy, "collect_email": req.collect_email,
            "call_link": req.book_call_link if req.book_call_active else "",
            "wa_num": req.whatsapp_number if req.whatsapp_active else "",
            "delay": str(req.response_delay_ms), "fallback": req.fallback_msg,
            "bot_personality": req.bot_personality, "bot_color": req.bot_color, "url": req.url
        }
        index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
        
        # 4. Start Background Crawl
        bg.add_task(deep_scraper_engine, req.url, req.client_id, final_api_key)
        return {"status": "success", "message": "Deep Sync Started."}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- UTILS & STATS ---
@app.post("/get-config")
async def get_conf(req: AutoSyncRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if res.vectors:
            d = res.vectors[f"config_{req.client_id}"].metadata
            return {"bot_name": d.get("bot_name"), "bot_color": d.get("bot_color"), "welcome_msg": f"Hi! I'm {d.get('bot_name')}. How can I help?"}
        return {"bot_name": "Support", "bot_color": "#4F46E5"}
    except: return {"bot_name": "Support", "bot_color": "#4F46E5"}

@app.post("/get-stats")
def stats_engine(req: AutoSyncRequest):
    try:
        dummy=[0.0]*768
        c = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "chat_log"})
        l = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "lead"})
        return {"visitors": len(c.get('matches',[])), "chats": len(c.get('matches',[])), "leads": len(l.get('matches',[]))}
    except: return {"visitors":0,"chats":0,"leads":0}

@app.post("/get-leads")
def leads_engine(req: AutoSyncRequest):
    try:
        res = index.query(namespace=req.client_id, vector=[0.0]*768, top_k=100, filter={"type": "lead"}, include_metadata=True)
        return {"leads": [{"email": m['metadata'].get('email'), "message": m['metadata'].get('user'), "date": m['metadata'].get('timestamp')} for m in res.get('matches',[])]}
    except: return {"leads": []}

@app.post("/verify-install")
async def verify_engine(req: AutoSyncRequest):
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(req.url if "http" in req.url else f"https://{req.url}", timeout=10, ssl=False) as r:
                return {"status": "success" if "widget.js" in (await r.text()) and req.client_id in (await r.text()) else "failed"}
    except: return {"status": "failed"}

@app.get("/")
def health(): return {"status": "Omni-Brain v13.0 SaaS Active"}
