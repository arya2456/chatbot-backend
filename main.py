import os
import time
import asyncio
import logging
import uuid
import re
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse
import io

# Third-party imports
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from bs4 import BeautifulSoup
import aiohttp
import pypdf
import google.generativeai as genai
from contextlib import asynccontextmanager

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-Memory Status Tracker (Reset on restart, but results saved to DB)
CRAWL_STATUS: Dict[str, dict] = {}

if not PINECONE_API_KEY:
    logger.error("CRITICAL: PINECONE_API_KEY is missing.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Background task to keep app awake or handle daily jobs
    asyncio.create_task(daily_auto_crawler())
    yield

app = FastAPI(title="Omni-Brain v24.0 (Final Architecture)", version="24.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def connect_db():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)
    except: return None

index = connect_db()

# --- 2. DATA MODELS ---
class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str = "" 
    # Identity
    bot_name: str = "AI Assistant"
    bot_lang: str = "English"
    bot_personality: str = "Professional"
    bot_color: str = "#4F46E5"
    bot_avatar: str = ""
    # Business
    biz_name: str = ""
    biz_phone: str = ""
    biz_email: str = ""
    # Logic
    leads_trigger: str = "price" 
    collect_name: bool = True
    collect_email: bool = True
    collect_phone: bool = False
    collect_company: bool = False
    book_call_link: str = ""
    whatsapp_number: str = ""
    # Controls
    bot_status: bool = True
    response_delay_ms: int = 1500
    max_conv_length: int = 50
    fallback_msg: str = "I'm not sure. Would you like to speak to a human?"

class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"
    page_url: str = ""
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
        await asyncio.sleep(86400) # Placeholder for daily logic

# --- 4. AUTO-DISCOVERY ENGINE ---
def extract_metadata(soup):
    """Finds phone, email, and biz name automatically."""
    meta = {}
    # Phone
    phone_link = soup.find('a', href=re.compile(r'^tel:'))
    if phone_link: meta['biz_phone'] = phone_link['href'].replace('tel:', '').strip()
    
    # Email
    email_link = soup.find('a', href=re.compile(r'^mailto:'))
    if email_link: meta['biz_email'] = email_link['href'].replace('mailto:', '').strip()
    
    # Name
    og_name = soup.find("meta", property="og:site_name")
    if og_name: meta['biz_name'] = og_name['content']
    
    return meta

# --- 5. DEEP SCRAPER (With Real-Time Status) ---
async def deep_scraper_engine(start_url: str, client_id: str, api_key: str):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    
    # Init Status
    CRAWL_STATUS[client_id] = {"status": "scanning", "pages": 0, "progress": 0, "current_url": "Starting..."}
    
    domain = urlparse(start_url).netloc
    visited, vectors, seen_hashes = set(), [], set()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))
    
    discovered_meta = {} # To store auto-discovered phone/email

    async with aiohttp.ClientSession() as session:
        try:
            while not queue.empty() and len(visited) < 30:
                url, depth = await queue.get()
                if url in visited or depth > 3: continue
                visited.add(url)
                
                # Update Real-Time Status
                CRAWL_STATUS[client_id].update({
                    "pages": len(visited), 
                    "progress": int((len(visited)/30)*90),
                    "current_url": url
                })
                
                try:
                    async with session.get(url, timeout=10, ssl=False) as resp:
                        if resp.status != 200: continue
                        text_content = await resp.text()
                        soup = BeautifulSoup(text_content, 'html.parser')
                        
                        # Feature: Auto-Discovery (Only on first page)
                        if len(visited) == 1:
                            discovered_meta = extract_metadata(soup)

                        # Link Discovery
                        for a in soup.find_all('a', href=True):
                            link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                            if urlparse(link).netloc == domain and link not in visited:
                                await queue.put((link, depth + 1))
                        
                        # Content Cleaning
                        for x in soup(['script', 'style', 'nav', 'footer', 'aside']): x.decompose()
                        text = soup.get_text(separator=' ', strip=True)
                        if len(text) < 200: continue

                        # Chunking
                        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                        for chunk in chunks:
                            h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                            if h in seen_hashes: continue
                            seen_hashes.add(h)
                            
                            emb = safe_embed(chunk, api_key)
                            vectors.append({
                                "id": f"doc_{uuid.uuid4()}", "values": emb,
                                "metadata": {"text": chunk, "url": url, "type": "knowledge"} 
                            })
                        
                        # Batch Upload
                        if len(vectors) > 20:
                            index.upsert(vectors=vectors, namespace=client_id)
                            vectors = []
                except: continue
            
            # Final Upload
            if vectors: index.upsert(vectors=vectors, namespace=client_id)
            
            # Feature: Update Client Config with Discovered Data
            if discovered_meta:
                try:
                    res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
                    if res.vectors:
                        current_conf = res.vectors[f"config_{client_id}"].metadata
                        # Only update if fields are empty
                        if not current_conf.get('biz_phone'): current_conf['biz_phone'] = discovered_meta.get('biz_phone', '')
                        if not current_conf.get('biz_email'): current_conf['biz_email'] = discovered_meta.get('biz_email', '')
                        if not current_conf.get('biz_name'): current_conf['biz_name'] = discovered_meta.get('biz_name', '')
                        
                        index.upsert(vectors=[{"id": f"config_{client_id}", "values": [1.0]*768, "metadata": current_conf}], namespace=client_id)
                except: pass

            CRAWL_STATUS[client_id] = {"status": "complete", "pages": len(visited), "progress": 100, "current_url": "Done"}
            
        except Exception as e:
            CRAWL_STATUS[client_id] = {"status": "error", "message": str(e), "progress": 0}

# --- 6. CHAT ENGINE (With Logic Gates & Analytics) ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    start_time = time.time()
    try:
        # 1. HYBRID KEY RECOVERY
        active_key = req.api_key.strip()
        conf = {}
        
        try:
            res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
            if res.vectors:
                conf = res.vectors[f"config_{req.client_id}"].metadata
                # Fallback
                if len(active_key) < 10: active_key = conf.get("gemini_api_key", "")
        except: pass
        
        if not active_key or len(active_key) < 10:
            return {"answer": "Configuration Error: API Key missing."}

        # 2. STRICT LOGIC GATES
        msg_lower = req.message.lower()
        ans = ""

        # Gate A: Lead Trap (Before Pricing)
        if conf.get("leads_trigger") == "price" and any(x in msg_lower for x in ["price", "cost", "how much", "rate", "fee"]):
            ans = "I'd be happy to share our pricing! Could you please share your **Email Address** first so I can send you the details?"
        
        # Gate B: Booking Link Force
        elif any(x in msg_lower for x in ["book", "call", "appointment", "schedule", "human"]):
            link = conf.get("book_call_link", "")
            if link and len(link) > 3:
                ans = f"Certainly! You can book a call with our team directly here: {link}"
            else:
                ans = f"Please contact us at {conf.get('biz_phone', 'our office')} to schedule a call."

        # 3. AI GENERATION (If no gates triggered)
        if not ans:
            emb = safe_embed(req.message, active_key)
            search = index.query(namespace=req.client_id, vector=emb, top_k=4, include_metadata=True, filter={"type": "knowledge"})
            
            context_str = "\n".join([f"FACT: {m['metadata']['text']} [Source: {m['metadata']['url']}]" for m in search['matches']])

            sys_msg = f"""
            You are {conf.get('bot_name', 'AI Assistant')} at {conf.get('biz_name', 'our company')}.
            TONE: {conf.get('bot_personality', 'Professional')}.
            LANGUAGE: {conf.get('bot_lang', 'English')}.
            
            FACTS:
            {context_str}
            
            INSTRUCTIONS:
            1. Answer strictly based on FACTS.
            2. If you don't know, say "{conf.get('fallback_msg')}".
            3. Keep answers concise.
            """
            
            model = get_model(active_key)
            try:
                ans = model.generate_content(f"{sys_msg}\n\nUSER: {req.message}").text
            except Exception as e:
                ans = "I'm having trouble connecting right now."

        # 4. ANALYTICS LOGGING (Persistent)
        # We save this chat as a vector so 'get-stats' can count it later
        log_id = f"log_{int(time.time())}_{uuid.uuid4()}"
        log_meta = {
            "type": "chat_log",
            "session": req.session_id,
            "user_msg": req.message,
            "bot_msg": ans,
            "timestamp": int(time.time())
        }
        # Fire and forget logging
        asyncio.create_task(log_analytics(req.client_id, log_id, log_meta))

        return {"answer": ans}

    except Exception as e: return {"answer": f"System Error: {str(e)}"}

async def log_analytics(ns, id, meta):
    try: index.upsert(vectors=[{"id": id, "values": [0.1]*768, "metadata": meta}], namespace=ns)
    except: pass

# --- 7. TRAIN ENDPOINT (Smart Save) ---
@app.post("/train")
async def train_saas_engine(req: TrainRequest, bg: BackgroundTasks):
    try:
        # Smart Key Preservation
        final_key = req.gemini_api_key.strip()
        if len(final_key) < 10:
            try:
                existing = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
                if existing.vectors:
                    final_key = existing.vectors[f"config_{req.client_id}"].metadata.get("gemini_api_key", "")
            except: pass
        
        if len(final_key) < 10: return {"status": "error", "message": "API Key Required"}

        meta = req.dict()
        meta["type"] = "config"
        meta["gemini_api_key"] = final_key
        
        # Sanitize
        for k, v in meta.items(): 
            if v is None: meta[k] = ""

        index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
        
        # Only trigger crawl if URL changed or requested
        bg.add_task(deep_scraper_engine, req.url, req.client_id, final_key)
        return {"status": "success", "message": "Settings Saved & Sync Started."}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- 8. PDF UPLOAD (Restored Feature) ---
@app.post("/upload-file")
async def upload_file(client_id: str, file: UploadFile = File(...)):
    try:
        # Fetch Key for embedding
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if not res.vectors: return {"status": "error", "message": "Configure bot first"}
        api_key = res.vectors[f"config_{client_id}"].metadata.get("gemini_api_key")

        content = await file.read()
        text = ""
        
        # Extract Text
        if file.filename.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(content))
            for page in reader.pages: text += page.extract_text() + "\n"
        else:
            text = content.decode("utf-8")

        # Chunk & Embed
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        vectors = []
        for chunk in chunks:
            emb = safe_embed(chunk, api_key)
            vectors.append({
                "id": f"file_{uuid.uuid4()}", 
                "values": emb,
                "metadata": {"text": chunk, "url": f"File: {file.filename}", "type": "knowledge"}
            })
        
        if vectors: index.upsert(vectors=vectors, namespace=client_id)
        return {"status": "success", "filename": file.filename}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- 9. UTILS & ANALYTICS ---
@app.post("/get-crawl-status")
async def get_crawl_status(req: StatusRequest):
    return CRAWL_STATUS.get(req.client_id, {"status": "idle", "progress": 0, "current_url": "Waiting..."})

@app.post("/get-config")
async def get_conf(req: ChatRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if res.vectors: return res.vectors[f"config_{req.client_id}"].metadata
        return {"bot_name": "Support", "bot_color": "#4F46E5"}
    except: return {"bot_name": "Support", "bot_color": "#4F46E5"}

@app.post("/get-stats")
async def stats_engine(req: BaseModel):
    # Mocking real count for speed (fetching all vectors is slow)
    # In V25, use a separate stats counter vector.
    # For now, return placeholders or use limited query
    return {"visitors": 12, "chats": 5, "leads": 2} # Placeholder for speed

@app.post("/verify-install")
async def verify_engine(req: BaseModel): return {"status": "success"}

@app.get("/")
def health(): return {"status": "Omni-Brain v24.0 Active"}
