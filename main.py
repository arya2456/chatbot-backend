import os, time, asyncio, logging, uuid, re, json, hashlib, io, requests
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from bs4 import BeautifulSoup
import aiohttp, pypdf
import google.generativeai as genai
from contextlib import asynccontextmanager

# --- 1. CONFIG ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")
# This points the AI to your new MySQL Brain!
PHP_DASHBOARD_URL = "https://dashboard.fcmedia.in/api.php" 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Omni-Brain v27.2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def connect_db():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)
    except: return None

index = connect_db()

# --- 2. MODELS ---
class TrainRequest(BaseModel):
    client_id: str; url: str; gemini_api_key: str = ""
    bot_name: str = "AI Assistant"; bot_lang: str = "English"; bot_personality: str = "Professional"
    bot_color: str = "#4F46E5"; bot_avatar: str = ""; biz_name: str = ""; biz_phone: str = ""; biz_email: str = ""
    leads_trigger: str = "price"; collect_name: bool = True; collect_email: bool = True
    collect_phone: bool = False; collect_company: bool = False; book_call_link: str = ""
    whatsapp_number: str = ""; bot_status: bool = True; response_delay_ms: int = 1500
    max_conv_length: int = 50; fallback_msg: str = "I'm not sure. Would you like to speak to a human?"

class ChatRequest(BaseModel):
    message: str; client_id: str; session_id: str = "Guest"; page_url: str = ""; api_key: str = ""

class StatusRequest(BaseModel):
    client_id: str; message: Optional[str] = None; session_id: Optional[str] = None
    api_key: Optional[str] = None; page_url: Optional[str] = None

# --- 3. HELPERS ---
def get_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

def safe_embed(text, api_key):
    genai.configure(api_key=api_key)
    try: return genai.embed_content(model="models/embedding-001", content=text)['embedding']
    except: return [0.0] * 768

# --- 4. ENGINE ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    try:
        # 1. Get Config & Context
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        conf = res.vectors[f"config_{req.client_id}"].metadata if res.vectors else {}
        active_key = req.api_key if len(req.api_key) > 10 else conf.get("gemini_api_key", "")
        if not active_key: return {"answer": "API Key missing."}

        emb = safe_embed(req.message, active_key)
        search = index.query(namespace=req.client_id, vector=emb, top_k=3, include_metadata=True, filter={"type": "knowledge"})
        context = "\n".join([m['metadata']['text'] for m in search['matches']])
        
        # 2. Generate AI Answer
        sys_msg = f"Role: {conf.get('bot_name')}. Language: {conf.get('bot_lang')}. Context: {context}. User: {req.message}"
        ans = get_model(active_key).generate_content(sys_msg).text
        
        # 3. Save to Pinecone (for AI context memory)
        log_meta = {"type": "chat_log", "session": req.session_id, "user_msg": req.message, "bot_msg": ans, "timestamp": int(time.time())}
        index.upsert(vectors=[{"id": f"log_{uuid.uuid4()}", "values": [0.1]*768, "metadata": log_meta}], namespace=req.client_id)
        
        # 4. PUSH TO MYSQL BRAIN (For Dashboard Analytics)
        try:
            payload = {
                "client_id": req.client_id,
                "session_id": req.session_id,
                "user_msg": req.message,
                "bot_msg": ans
            }
            # Using synchronous requests for guaranteed delivery
            db_response = requests.post(f"{PHP_DASHBOARD_URL}?action=save_chat", json=payload, timeout=3)
            logger.info(f"MySQL Sync Status: {db_response.status_code}")
        except Exception as db_err:
            logger.error(f"FATAL: Could not sync to MySQL: {db_err}")

        return {"answer": ans}
    except Exception as e: return {"answer": f"System error: {str(e)}"}

# --- THE NEW LEAD PUSHER ---
@app.post("/capture-lead")
async def capture_lead(req: dict):
    # If the widget sends leads here, immediately push them to the MySQL database
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(f"{PHP_DASHBOARD_URL}?action=save_lead", json=req)
        
        # Optional: Save a backup to Pinecone just in case
        log_meta = req
        log_meta["type"] = "lead"
        log_meta["timestamp"] = int(time.time())
        index.upsert(vectors=[{"id": f"lead_{uuid.uuid4()}", "values": [0.0]*768, "metadata": log_meta}], namespace=req.get("client_id", "default"))
        
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/get-config")
async def get_conf(req: StatusRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if res.vectors: return res.vectors[f"config_{req.client_id}"].metadata
        return {"bot_name": "Support", "bot_color": "#4F46E5"}
    except: return {"bot_name": "Support", "bot_color": "#4F46E5"}

@app.post("/train")
async def train_saas_engine(req: TrainRequest):
    meta = req.dict(); meta["type"] = "config"
    index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
    return {"status": "success"}

@app.get("/")
def health(): return {"status": "Omni-Brain v27.2 Active - Synced to MySQL"}
