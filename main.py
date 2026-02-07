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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not PINECONE_API_KEY:
    logger.error("CRITICAL: PINECONE_API_KEY is missing.")

app = FastAPI(title="Omni-Brain v14.2 (Auto-Detect)", version="14.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- DATABASE CONNECTION ---
def connect_db():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"❌ DB Connection Failed: {e}")
        return None

index = connect_db()

# --- DYNAMIC MODEL FINDER (THE FIX) ---
# This function asks Google: "What embedding model can I use?"
def get_optimal_models(api_key):
    try:
        genai.configure(api_key=api_key)
        found_embed = None
        
        # 1. List all models and find one that supports 'embedContent'
        for m in genai.list_models():
            if 'embedContent' in m.supported_generation_methods:
                found_embed = m.name
                # Prefer the newest 004 model if available
                if 'text-embedding-004' in m.name:
                    break 
        
        # Default fallback if list fails
        if not found_embed: found_embed = "models/text-embedding-004"
        
        return {
            "chat": "gemini-2.0-flash", # We know you have this
            "embed": found_embed
        }
    except:
        return {"chat": "gemini-2.0-flash", "embed": "models/text-embedding-004"}

# --- DATA MODELS ---
class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str = "" 
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
    bot_avatar: str = ""

class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"
    page_url: str = ""

class AutoSyncRequest(BaseModel):
    client_id: str
    url: str = ""

# --- HELPER: GENAI RETRY ---
async def generate_answer_with_retry(model, prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            return model.generate_content(prompt).text
        except Exception as e:
            if attempt == retries: raise
            await asyncio.sleep(0.5 * (attempt + 1))

# --- HELPER: MEMORY ---
def get_conversation_history(client_id: str, session_id: str, limit: int = 6):
    try:
        dummy = [0.0] * 768
        res = index.query(
            namespace=client_id, vector=dummy, top_k=limit, 
            filter={"type": "chat_log", "session": session_id}, 
            include_metadata=True
        )
        matches = sorted(res.get("matches", []), key=lambda m: m["metadata"].get("timestamp", 0))
        return "\n".join([f"User: {m['metadata'].get('user','')}\nAI: {m['metadata'].get('bot','')}" for m in matches])
    except: return ""

# --- HELPER: THEME ---
async def extract_theme_color(session, url):
    try:
        async with session.get(url, timeout=10, ssl=False) as resp:
            if resp.status != 200: return "#4F46E5"
            soup = BeautifulSoup(await resp.text(), 'html.parser')
            meta = soup.find("meta", {"name": "theme-color"})
            return meta.get("content") if meta else "#4F46E5"
    except: return "#4F46E5"

# --- STEP 1: SCRAPER ---
async def deep_scraper_engine(start_url: str, client_id: str, api_key: str, max_pages: int = 40):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    
    # Auto-Detect Models
    models = get_optimal_models(api_key)
    
    domain = urlparse(start_url).netloc
    visited, vectors, seen_hashes = set(), [], set()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))

    async with aiohttp.ClientSession() as session:
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
                    for a in soup.find_all('a', href=True):
                        link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                        if urlparse(link).netloc == domain and link not in visited:
                            await queue.put((link, depth + 1))
                    for x in soup(['script', 'style', 'nav', 'footer', 'iframe', 'aside']): x.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    if len(text) < 200: continue

                    cleaner = genai.GenerativeModel(models["chat"])
                    clean_text = await generate_answer_with_retry(cleaner, f"Extract business facts only. Remove fluff. TEXT: {text[:8000]}")

                    chunks = [clean_text[i:i+1500] for i in range(0, len(clean_text), 1500)]
                    for i, chunk in enumerate(chunks):
                        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                        if h in seen_hashes: continue
                        seen_hashes.add(h)

                        # AUTO-DETECTED MODEL USED HERE
                        emb = genai.embed_content(model=models["embed"], content=chunk)['embedding']
                        vectors.append({
                            "id": f"neural_{uuid.uuid4()}",
                            "values": emb,
                            "metadata": {"text": chunk, "url": url, "type": "knowledge", "hash": h, "source": "recursive_crawl"}
                        })
                    
                    if len(vectors) > 30:
                        index.upsert(vectors=vectors, namespace=client_id)
                        vectors = []
            except: continue
    if vectors: index.upsert(vectors=vectors, namespace=client_id)

# --- UPLOAD ---
@app.post("/upload-file")
async def upload_file_engine(client_id: str, file: UploadFile = File(...)):
    try:
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if not res.vectors: return {"status": "error", "message": "Train website first."}
        
        api_key = res.vectors[f"config_{client_id}"].metadata.get('api_key')
        if not api_key: return {"status": "error", "message": "API Key missing."}
        
        models = get_optimal_models(api_key)

        content = ""
        if file.content_type == "application/pdf":
            reader = pypdf.PdfReader(io.BytesIO(await file.read()))
            for p in reader.pages: content += p.extract_text() + "\n"
        else: content = (await file.read()).decode("utf-8")

        chunks = [content[i:i+1200] for i in range(0, len(content), 1200)]
        vectors = []
        for i, c in enumerate(chunks):
             emb = genai.embed_content(model=models["embed"], content=c)['embedding']
             vectors.append({
                 "id": f"doc_{uuid.uuid4()}",
                 "values": emb,
                 "metadata": {"text": c, "source": file.filename, "type": "knowledge", "category": "uploaded_document"}
             })
        index.upsert(vectors=vectors, namespace=client_id)
        return {"status": "success", "message": f"Learned from {file.filename}"}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- CHAT ENGINE (AUTO-DETECT) ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    try:
        if index is None: return {"answer": "Critical: Database Error."}
        try: res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        except: return {"answer": "Connection Error. Retrying..."}

        if not res.vectors: return {"answer": "Brain not active. Please click 'Save' in dashboard."}
        
        conf = res.vectors[f"config_{req.client_id}"].metadata
        
        if str(conf.get("bot_status", "True")).lower() in ("false", "0", "off"):
            return {"answer": "This assistant is currently offline."}

        api_key = conf.get('api_key', '').strip()
        if not api_key: return {"answer": "Error: Admin must configure API Key."}
        
        # 1. AUTO-DETECT MODELS HERE
        models = get_optimal_models(api_key)

        await asyncio.sleep(int(conf.get("delay", 1000)) / 1000)

        if conf.get("leads_trigger") == "Before sharing pricing" and conf.get("collect_email", True):
             if any(x in req.message.lower() for x in ["price", "cost", "how much", "fees"]):
                 dummy = [0.0]*768
                 ex = index.query(namespace=req.client_id, vector=dummy, top_k=1, filter={"type": "lead", "session": req.session_id})
                 if not ex.get("matches"):
                     return {"answer": "I'd be happy to share pricing! Could you please share your email address first?"}

        history = get_conversation_history(req.client_id, req.session_id)
        
        # 2. USE AUTO-DETECTED MODEL
        try:
            emb = genai.embed_content(model=models["embed"], content=req.message)['embedding']
        except Exception as e:
            return {"answer": f"Embedding Error: {str(e)}"}

        search = index.query(namespace=req.client_id, vector=emb, top_k=6, include_metadata=True, filter={"type": "knowledge"})
        ctx = "\n\n".join([m['metadata']['text'] for m in search['matches']])

        sys_msg = f"""
        IDENTITY: You are {conf.get('bot_name')} at "{conf.get('biz_name')}". 
        LANGUAGE: Answer strictly in {conf.get('bot_lang', 'English')}.
        RULES: 1-3 lines max. No hallucinations. Use HISTORY.
        KNOWLEDGE: {ctx}
        HISTORY: {history}
        """
        
        model = genai.GenerativeModel(models["chat"])
        ans = await generate_answer_with_retry(model, f"{sys_msg}\n\nUSER: {req.message}")

        log_id = f"log_{req.session_id}_{uuid.uuid4()}"
        m_type = "lead" if re.search(r"[\w\.-]+@[\w\.-]+", req.message) else "chat_log"
        meta = {"type": m_type, "user": req.message, "bot": ans, "session": req.session_id, "timestamp": int(time.time()), "email": req.message if m_type == "lead" else ""}
        index.upsert(vectors=[{"id": log_id, "values": [0.1]*768, "metadata": meta}], namespace=req.client_id)
        
        return {"answer": ans}
    except Exception as e:
        logger.error(f"CHAT ERROR: {e}")
        return {"answer": f"System Error: {str(e)}"}

# --- TRAIN ---
@app.post("/train")
async def train_saas_engine(req: TrainRequest, bg: BackgroundTasks):
    try:
        final_api_key = req.gemini_api_key.strip()
        if not final_api_key:
            existing = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
            if existing.vectors:
                final_api_key = existing.vectors[f"config_{req.client_id}"].metadata.get("api_key")
        
        if not final_api_key: return {"status": "error", "message": "No API Key found."}

        try: index.delete(delete_all=True, namespace=req.client_id)
        except: pass

        meta = {
            "type": "config", "api_key": final_api_key, 
            "bot_name": req.bot_name, "bot_lang": req.bot_lang, 
            "bot_status": str(req.bot_status), "biz_name": req.biz_name, 
            "biz_phone": req.biz_phone, "biz_email": req.biz_email,
            "leads_trigger": req.trigger_strategy, "collect_email": req.collect_email,
            "call_link": req.book_call_link, "wa_num": req.whatsapp_number,
            "delay": str(req.response_delay_ms), "fallback": req.fallback_msg,
            "bot_personality": req.bot_personality, "bot_color": req.bot_color, "url": req.url,
            "bot_avatar": req.bot_avatar
        }
        index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
        
        bg.add_task(deep_scraper_engine, req.url, req.client_id, final_api_key)
        return {"status": "success", "message": "Deep Sync Started."}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- UTILS ---
@app.post("/get-config")
async def get_conf(req: AutoSyncRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if res.vectors:
            d = res.vectors[f"config_{req.client_id}"].metadata
            return {
                "bot_name": d.get("bot_name"), "bot_color": d.get("bot_color"),
                "bot_avatar": d.get("bot_avatar", ""),
                "welcome_msg": f"Hi! I'm {d.get('bot_name')}. How can I help?"
            }
        return {"bot_name": "Support", "bot_color": "#4F46E5", "bot_avatar": ""}
    except: return {"bot_name": "Support", "bot_color": "#4F46E5", "bot_avatar": ""}

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
def health(): return {"status": "Omni-Brain v14.2 Auto-Detect Active"}
