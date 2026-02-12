import os
import time
import asyncio
import logging
import uuid
import re
import json 
import hashlib # <--- NEW: Added to prevent Crawler Crash
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

app = FastAPI(title="Omni-Brain v26.4 (Production Stable)", version="26.4", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def connect_db():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)
    except: return None

index = connect_db()

# --- SAFETY CHECK: Ensure Database is Connected ---
if index is None:
    logger.error("CRITICAL: Pinecone Index Connection Failed. Check API Key.")

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

# --- STATUS REQUEST (Universal Key for Dashboard & Widget Handshakes) ---
class StatusRequest(BaseModel):
    client_id: str
    # Added these optional fields so the backend doesn't reject the widget's connection
    message: Optional[str] = None
    session_id: Optional[str] = None
    api_key: Optional[str] = None
    page_url: Optional[str] = None

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

# --- 4. COGNITIVE ENGINE (INTENT & EXTRACTION) ---
async def classify_intent_and_extract(message: str, history: str, api_key: str):
    """
    Identifies User Intent and Extracts Leads in one fast step.
    Returns JSON: { "intent": "BUY"|"SUPPORT"|"GENERAL", "email": "...", "name": "..." }
    """
    try:
        model = get_model(api_key)
        prompt = f"""
        ANALYZE this message in the context of a conversation.
        
        HISTORY:
        {history}
        
        MESSAGE: "{message}"
        
        TASK:
        1. Classify INTENT: 
           - 'BOOKING' (wants to call/book/appointment/talk to human)
           - 'PRICE' (asking cost/rates)
           - 'INFO' (general questions/blog/website)
           - 'GREETING' (hi/hello)
           - 'AGREEMENT' (yes/ok/please - implying agreement to previous bot offer)
           - 'OTHER'
        2. Extract ENTITIES if present (Name, Email, Phone). Return null if not found.
        
        OUTPUT JSON ONLY:
        {{ "intent": "...", "name": null, "email": null, "phone": null }}
        """
        res = model.generate_content(prompt)
        cleaned = res.text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except:
        return {"intent": "OTHER", "name": None, "email": None, "phone": None}

def get_session_history_list(client_id: str, session_id: str, limit: int = 4):
    try:
        dummy = [0.0] * 768
        res = index.query(
            namespace=client_id,
            vector=dummy,
            filter={"type": "chat_log", "session": session_id},
            top_k=limit,
            include_metadata=True
        )
        matches = sorted(res['matches'], key=lambda x: x['metadata'].get('timestamp', 0))
        return [{"user": m['metadata'].get('user_msg',''), "bot": m['metadata'].get('bot_msg','')} for m in matches]
    except: return []

def format_history_str(history_list):
    return "\n".join([f"User: {x['user']}\nBot: {x['bot']}" for x in history_list])

def extract_metadata(soup):
    """Finds phone, email, and biz name automatically."""
    meta = {}
    phone_link = soup.find('a', href=re.compile(r'^tel:'))
    if phone_link: meta['biz_phone'] = phone_link['href'].replace('tel:', '').strip()
    
    email_link = soup.find('a', href=re.compile(r'^mailto:'))
    if email_link: meta['biz_email'] = email_link['href'].replace('mailto:', '').strip()
    
    og_name = soup.find("meta", property="og:site_name")
    if og_name: meta['biz_name'] = og_name['content']
    return meta

# --- 5. DEEP SCRAPER (With Real-Time Status) ---
async def deep_scraper_engine(start_url: str, client_id: str, api_key: str):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    
    CRAWL_STATUS[client_id] = {"status": "scanning", "pages": 0, "progress": 0, "current_url": "Starting..."}
    
    domain = urlparse(start_url).netloc
    visited, vectors, seen_hashes = set(), [], set()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))
    
    discovered_meta = {} 

    async with aiohttp.ClientSession() as session:
        try:
            while not queue.empty() and len(visited) < 30:
                url, depth = await queue.get()
                if url in visited or depth > 3: continue
                visited.add(url)
                
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
                        
                        if len(visited) == 1:
                            discovered_meta = extract_metadata(soup)

                        for a in soup.find_all('a', href=True):
                            link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                            if urlparse(link).netloc == domain and link not in visited:
                                await queue.put((link, depth + 1))
                        
                        for x in soup(['script', 'style', 'nav', 'footer', 'aside']): x.decompose()
                        text = soup.get_text(separator=' ', strip=True)
                        if len(text) < 200: continue

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
                        
                        if len(vectors) > 20:
                            index.upsert(vectors=vectors, namespace=client_id)
                            vectors = []
                except: continue
            
            if vectors: index.upsert(vectors=vectors, namespace=client_id)
            
            if discovered_meta:
                try:
                    res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
                    if res.vectors:
                        current_conf = res.vectors[f"config_{client_id}"].metadata
                        updated = False
                        if not current_conf.get('biz_phone') and discovered_meta.get('biz_phone'): 
                            current_conf['biz_phone'] = discovered_meta['biz_phone']; updated=True
                        if not current_conf.get('biz_email') and discovered_meta.get('biz_email'): 
                            current_conf['biz_email'] = discovered_meta['biz_email']; updated=True
                        if updated:
                            index.upsert(vectors=[{"id": f"config_{client_id}", "values": [1.0]*768, "metadata": current_conf}], namespace=client_id)
                except: pass

            CRAWL_STATUS[client_id] = {"status": "complete", "pages": len(visited), "progress": 100, "current_url": "Done"}
            
        except Exception as e:
            CRAWL_STATUS[client_id] = {"status": "error", "message": str(e), "progress": 0}

# --- 6. CHAT ENGINE (UPGRADED: COGNITIVE + SMART GATES) ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    try:
        # 1. SETUP & KEY RECOVERY
        active_key = req.api_key.strip()
        conf = {}
        try:
            res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
            if res.vectors:
                conf = res.vectors[f"config_{req.client_id}"].metadata
                if len(active_key) < 10: active_key = conf.get("gemini_api_key", "")
        except: pass
        
        if not active_key or len(active_key) < 10: return {"answer": "Error: API Key missing."}

        # 2. CONTEXT & INTENT
        history_list = get_session_history_list(req.client_id, req.session_id)
        history_str = format_history_str(history_list)
        
        # COGNITIVE JUMP: Classify Intent using AI
        cognition = await classify_intent_and_extract(req.message, history_str, active_key)
        intent = cognition.get("intent", "OTHER")
        
        # AUTO-CAPTURE LEADS (Natural Slot Filling)
        if cognition.get("email") or cognition.get("phone"):
            lead_id = f"lead_{int(time.time())}"
            lead_meta = {
                "type": "lead", 
                "email": cognition.get("email"), 
                "phone": cognition.get("phone"), 
                "name": cognition.get("name"), 
                "message": req.message, 
                "date": int(time.time())
            }
            asyncio.create_task(log_analytics(req.client_id, lead_id, lead_meta))

        # 3. LOGIC GATES (Driven by Intent)
        ans = ""
        
        # Define Contact Info Helpers (Safe Fallback Logic)
        link = conf.get("book_call_link", "")
        phone = conf.get("biz_phone", "")
        email = conf.get("biz_email", "our support team")

        # Logic A: User Agrees to "Human/Call" offer
        last_bot_msg = history_list[-1]['bot'].lower() if history_list else ""
        was_offering = any(x in last_bot_msg for x in ["speak", "connect", "schedule", "call"])
        
        if intent == "AGREEMENT" and was_offering:
             if link and len(link) > 3: ans = f"Great! You can book a slot here: {link}"
             elif phone and len(phone) > 3: ans = f"Please give us a call at {phone}."
             else: ans = f"Please email us at {email} and we will set that up."
        
        # Logic B: Intent is BOOKING or URGENT
        elif not ans and (intent == "BOOKING" or "fast" in req.message.lower()):
            if link and len(link) > 3: ans = f"Certainly! You can book a call here: {link}"
            elif phone and len(phone) > 3: ans = f"You can reach us immediately at {phone}."
            else: ans = f"We'd love to connect. Please contact us via email at {email}."

        # Logic C: Intent is PRICE (Lead Trap)
        elif not ans and intent == "PRICE" and conf.get("leads_trigger") == "price":
            # If we don't have email yet, trigger the trap
            if not cognition.get("email"):
                ans = "I'd be happy to share our pricing! Could you please share your **Email Address** first so I can send you the details?"

        # 4. RAG GENERATION
        if not ans:
            emb = safe_embed(req.message, active_key)
            search = index.query(namespace=req.client_id, vector=emb, top_k=4, include_metadata=True, filter={"type": "knowledge"})
            context = "\n".join([f"FACT: {m['metadata']['text']} [Source: {m['metadata'].get('url','')}]" for m in search['matches']])

            sys_msg = f"""
            ROLE: You are {conf.get('bot_name', 'AI Assistant')} for {conf.get('biz_name', 'this company')}.
            TONE: {conf.get('bot_personality', 'Professional')}.
            LANGUAGE INSTRUCTION: You MUST answer in {conf.get('bot_lang', 'English')}.
            INTENT DETECTED: {intent}
            
            KNOWLEDGE:
            {context}
            
            HISTORY:
            {history_str}
            
            INSTRUCTIONS:
            1. Answer the user based on INTENT and KNOWLEDGE.
            2. If INTENT is 'INFO' or 'OTHER', explain using facts.
            3. If user provided Email/Phone, acknowledge it ("Thanks for that info...").
            4. If KNOWLEDGE missing, say "{conf.get('fallback_msg')}".
            """
            model = get_model(active_key)
            try: ans = model.generate_content(f"{sys_msg}\n\nUSER: {req.message}").text
            except: ans = "I'm having trouble connecting."

        # 5. LOGGING
        log_id = f"log_{int(time.time())}_{uuid.uuid4()}"
        asyncio.create_task(log_analytics(req.client_id, log_id, {
            "type": "chat_log", "session": req.session_id, "user_msg": req.message, "bot_msg": ans, "timestamp": int(time.time())
        }))

        return {"answer": ans}

    except Exception as e: return {"answer": f"System Error: {str(e)}"}

async def log_analytics(ns, id, meta):
    try: index.upsert(vectors=[{"id": id, "values": [0.1]*768, "metadata": meta}], namespace=ns)
    except: pass

# --- 7. TRAIN ENDPOINT (Smart Save) ---
@app.post("/train")
async def train_saas_engine(req: TrainRequest, bg: BackgroundTasks):
    try:
        final_key = req.gemini_api_key.strip()
        if len(final_key) < 10:
            try:
                existing = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
                if existing.vectors: final_key = existing.vectors[f"config_{req.client_id}"].metadata.get("gemini_api_key", "")
            except: pass
        if len(final_key) < 10: return {"status": "error", "message": "API Key Required"}

        meta = req.dict()
        meta["type"] = "config"
        meta["gemini_api_key"] = final_key
        for k, v in meta.items(): 
            if v is None: meta[k] = ""
        index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
        if req.url: bg.add_task(deep_scraper_engine, req.url, req.client_id, final_key)
        return {"status": "success", "message": "Settings Saved & Sync Started."}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- 8. PDF UPLOAD (Restored Feature) ---
@app.post("/upload-file")
async def upload_file(client_id: str, file: UploadFile = File(...)):
    try:
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if not res.vectors: return {"status": "error", "message": "Configure bot first"}
        api_key = res.vectors[f"config_{client_id}"].metadata.get("gemini_api_key")

        content = await file.read()
        text = ""
        
        if file.filename.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(content))
            for page in reader.pages: text += page.extract_text() + "\n"
        else:
            text = content.decode("utf-8")

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

# --- 9. REAL ANALYTICS (FIXED: Accepts StatusRequest properly) ---
@app.post("/get-analytics")
async def analytics_engine(req: StatusRequest): # <--- FIXED
    try:
        dummy = [0.0] * 768
        res = index.query(
            namespace=req.client_id,
            vector=dummy,
            filter={"type": "chat_log"},
            top_k=50,
            include_metadata=True
        )
        logs = []
        for m in res['matches']:
            logs.append({
                "session": m['metadata'].get('session', 'Unknown'),
                "user": m['metadata'].get('user_msg', ''),
                "bot": m['metadata'].get('bot_msg', ''),
                "time": m['metadata'].get('timestamp', 0)
            })
        logs.sort(key=lambda x: x['time'], reverse=True)
        return {"logs": logs}
    except: return {"logs": []}

@app.post("/get-stats")
async def stats_engine(req: StatusRequest): # <--- FIXED
    """
    Fetches ACTUAL metrics from Pinecone (Persistent).
    """
    try:
        dummy = [0.0] * 768
        # 1. Count Unique Visitors (Sessions)
        chat_res = index.query(namespace=req.client_id, vector=dummy, filter={"type": "chat_log"}, top_k=10000, include_metadata=True)
        unique_sessions = set([m['metadata'].get('session') for m in chat_res['matches']])
        total_chats = len(chat_res['matches'])
        
        # 2. Count Leads
        lead_res = index.query(namespace=req.client_id, vector=dummy, filter={"type": "lead"}, top_k=10000, include_metadata=True)
        total_leads = len(lead_res['matches'])
        
        return {
            "visitors": len(unique_sessions),
            "chats": total_chats,
            "leads": total_leads
        }
    except:
        return {"visitors": 0, "chats": 0, "leads": 0}

@app.post("/get-leads")
def leads_engine(req: StatusRequest): # <--- FIXED
    """Fetches ACTUAL leads from Pinecone"""
    try:
        dummy = [0.0] * 768
        res = index.query(namespace=req.client_id, vector=dummy, filter={"type": "lead"}, top_k=100, include_metadata=True)
        leads = []
        for m in res['matches']:
            leads.append({
                "email": m['metadata'].get('email') or m['metadata'].get('phone') or "No Contact",
                "message": m['metadata'].get('message', ''),
                "date": m['metadata'].get('date', 0)
            })
        leads.sort(key=lambda x: x['date'], reverse=True)
        return {"leads": leads}
    except: return {"leads": []}

@app.post("/get-crawl-status")
async def get_crawl_status(req: StatusRequest):
    return CRAWL_STATUS.get(req.client_id, {"status": "idle", "progress": 0, "current_url": "Waiting..."})

# --- 10. CONFIG ENDPOINT (FIXED FOR AVATAR LOADING) ---
@app.post("/get-config")
async def get_conf(req: StatusRequest): # <--- FIXED
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if res.vectors: return res.vectors[f"config_{req.client_id}"].metadata
        return {"bot_name": "Support", "bot_color": "#4F46E5"}
    except: return {"bot_name": "Support", "bot_color": "#4F46E5"}

@app.post("/verify-install")
async def verify_engine(req: BaseModel): return {"status": "success"}

@app.get("/")
def health(): return {"status": "Omni-Brain v26.4 Active"}
