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

# --- 1. SYSTEM CONFIG ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")

if not PINECONE_API_KEY:
    raise RuntimeError("CRITICAL: PINECONE_API_KEY is missing.")

# Optional: Backend Auth for your dashboard to talk to this API
API_AUTH_KEY = os.getenv("BACKEND_API_KEY") 

def verify_api_key(x_api_key: str = Header(None)):
    if API_AUTH_KEY and x_api_key != API_AUTH_KEY:
        raise HTTPException(status_code=401, detail="Invalid Backend Key")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Omni-Brain v12.1 SaaS Mode", version="12.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- DATABASE INIT ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    logger.info("✅ Pinecone DB Connected")
except Exception as e:
    logger.error(f"❌ Database Connection Failed: {e}")
    index = None

# --- MODELS ---
class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str  # RESTORED: Client provides their own key
    bot_name: str = "AI Assistant"
    bot_lang: str = "English"
    timezone: str = "Auto-detect (IST)"
    bot_status: bool = True
    biz_name: str = ""
    biz_phone: str = ""
    biz_email: str = ""
    collect_email: bool = True
    trigger_strategy: str = "Before sharing pricing"
    book_call_link: str = ""
    response_delay_ms: int = 1000
    fallback_msg: str = "I'm not sure. Would you like to speak to a human?"
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

# --- HELPER: RETRY LOGIC ---
async def generate_answer_with_retry(model, prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            return model.generate_content(prompt).text
        except Exception as e:
            if attempt == retries: raise
            await asyncio.sleep(0.5 * (attempt + 1))

# --- HELPER: THEME DETECTION ---
async def extract_theme_color(session, url):
    try:
        async with session.get(url, timeout=10, ssl=False) as resp:
            if resp.status != 200: return "#4F46E5"
            soup = BeautifulSoup(await resp.text(), 'html.parser')
            meta = soup.find("meta", {"name": "theme-color"})
            return meta.get("content") if meta else "#4F46E5"
    except: return "#4F46E5"

# --- STEP 1: DEEP SAAS CRAWLER ---
async def deep_scraper_engine(start_url: str, client_id: str, api_key: str, max_pages: int = 40):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    
    # CONFIGURE CLIENT KEY
    genai.configure(api_key=api_key)
    
    domain = urlparse(start_url).netloc
    visited, vectors, seen_hashes = set(), [], set()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))

    async with aiohttp.ClientSession() as session:
        # Auto-Theme
        color = await extract_theme_color(session, start_url)
        try:
            # Update theme if default
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
                    
                    # Links
                    for a in soup.find_all('a', href=True):
                        link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                        if urlparse(link).netloc == domain and link not in visited:
                            await queue.put((link, depth + 1))

                    # Clean
                    for x in soup(['script', 'style', 'nav', 'footer', 'iframe']): x.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    if len(text) < 200: continue

                    # AI Cleaning
                    cleaner = genai.GenerativeModel("gemini-2.5-flash")
                    clean_text = await generate_answer_with_retry(cleaner, f"Extract business facts only. Remove fluff. TEXT: {text[:8000]}")

                    # Chunking
                    chunks = [clean_text[i:i+1500] for i in range(0, len(clean_text), 1500)]
                    for i, chunk in enumerate(chunks):
                        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                        if h in seen_hashes: continue
                        seen_hashes.add(h)

                        emb = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
                        vectors.append({
                            "id": f"neural_{uuid.uuid4()}",
                            "values": emb,
                            "metadata": {"text": chunk, "url": url, "type": "knowledge", "hash": h}
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
        # Fetch Client Key
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if not res.vectors: return {"status": "error", "message": "Train website first."}
        api_key = res.vectors[f"config_{client_id}"].metadata['api_key']
        genai.configure(api_key=api_key)

        content = ""
        if file.content_type == "application/pdf":
            reader = pypdf.PdfReader(io.BytesIO(await file.read()))
            for p in reader.pages: content += p.extract_text() + "\n"
        else: content = (await file.read()).decode("utf-8")

        chunks = [content[i:i+1200] for i in range(0, len(content), 1200)]
        vectors = [{"id": f"doc_{uuid.uuid4()}", "values": genai.embed_content(model="models/text-embedding-004", content=c)['embedding'], "metadata": {"text": c, "source": file.filename, "type": "knowledge"}} for i, c in enumerate(chunks)]
        index.upsert(vectors=vectors, namespace=client_id)
        return {"status": "success", "message": f"Learned from {file.filename}"}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- SAAS CHAT ENGINE (CLIENT KEYS RESTORED) ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    try:
        # 1. Fetch Config & API KEY
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if not res.vectors: return {"answer": "Initializing... please refresh."}
        conf = res.vectors[f"config_{req.client_id}"].metadata
        
        # 2. Status Check
        if str(conf.get("bot_status", "True")).lower() in ("false", "0", "off"):
            return {"answer": "This assistant is currently offline."}

        # 3. CONFIGURE CLIENT KEY (Critical for SaaS)
        if 'api_key' not in conf: return {"answer": "API Key missing. Please re-train bot."}
        genai.configure(api_key=conf['api_key'])

        await asyncio.sleep(int(conf.get("delay", 1000)) / 1000)

        # 4. Memory Fetch
        dummy = [0.0] * 768
        hist_res = index.query(namespace=req.client_id, vector=dummy, top_k=5, filter={"type": "chat_log", "session": req.session_id}, include_metadata=True)
        history = "\n".join(sorted([f"User: {m['metadata']['user']}\nAI: {m['metadata']['bot']}" for m in hist_res.get('matches', [])], key=lambda x: x[0]))

        # 5. Knowledge Search
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
        - If missing info: "{conf.get('fallback')}"
        
        KNOWLEDGE: {ctx}
        HISTORY: {history}
        """
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        ans = await generate_answer_with_retry(model, f"{sys_msg}\n\nUSER: {req.message}")

        # 7. Secure Logging
        log_id = f"log_{req.session_id}_{int(time.time()*1000)}"
        m_type = "lead" if re.search(r"[\w\.-]+@[\w\.-]+", req.message) else "chat_log"
        index.upsert(vectors=[{"id": log_id, "values": dummy, "metadata": {"type": m_type, "user": req.message, "bot": ans, "session": req.session_id, "timestamp": int(time.time()), "email": req.message if m_type=="lead" else ""}}], namespace=req.client_id)
        
        return {"answer": ans}
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return {"answer": "I'm optimizing my neural links. Try again?"}

# --- TRAIN (WIPE & SYNC + SAVE KEY) ---
@app.post("/train")
async def train_saas_engine(req: TrainRequest, bg: BackgroundTasks):
    try:
        # Wipe old data
        try: index.delete(delete_all=True, namespace=req.client_id)
        except: pass

        # Save config WITH API KEY
        meta = {
            "type": "config", 
            "api_key": req.gemini_api_key, # Stored here for SaaS usage
            "bot_name": req.bot_name, "bot_lang": req.bot_lang, 
            "bot_status": str(req.bot_status), "biz_name": req.biz_name, 
            "biz_phone": req.biz_phone, "biz_email": req.biz_email,
            "leads_trigger": req.trigger_strategy, "call_link": req.book_call_link,
            "delay": str(req.response_delay_ms), "fallback": req.fallback_msg,
            "bot_personality": req.bot_personality, "bot_color": req.bot_color, "url": req.url
        }
        index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
        
        bg.add_task(deep_scraper_engine, req.url, req.client_id, req.gemini_api_key)
        return {"status": "success", "message": "Deep Sync Started."}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- UTILS ---
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
        return {"visitors": len(c['matches']), "chats": len(c['matches']), "leads": len(l['matches'])}
    except: return {"visitors":0,"chats":0,"leads":0}

@app.post("/get-leads")
def leads_engine(req: AutoSyncRequest):
    res = index.query(namespace=req.client_id, vector=[0.0]*768, top_k=100, filter={"type": "lead"}, include_metadata=True)
    return {"leads": [{"email": m['metadata'].get('email'), "message": m['metadata'].get('user'), "date": m['metadata'].get('timestamp')} for m in res['matches']]}

@app.post("/verify-install")
async def verify_engine(req: AutoSyncRequest):
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(req.url if "http" in req.url else f"https://{req.url}", timeout=10, ssl=False) as r:
                return {"status": "success" if "widget.js" in (await r.text()) and req.client_id in (await r.text()) else "failed"}
    except: return {"status": "failed"}

@app.get("/")
def health(): return {"status": "Omni-Brain v12.1 SaaS Active"}
