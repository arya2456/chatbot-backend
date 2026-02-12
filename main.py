import os, time, asyncio, logging, uuid, re, json, hashlib, io
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
    # Using the specific version that works for your environment
    return genai.GenerativeModel("gemini-2.0-flash")

def safe_embed(text, api_key):
    genai.configure(api_key=api_key)
    try: return genai.embed_content(model="models/embedding-001", content=text)['embedding']
    except: return [0.0] * 768

# --- 4. ENGINE ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        conf = res.vectors[f"config_{req.client_id}"].metadata if res.vectors else {}
        active_key = req.api_key if len(req.api_key) > 10 else conf.get("gemini_api_key", "")
        if not active_key: return {"answer": "API Key missing."}

        emb = safe_embed(req.message, active_key)
        search = index.query(namespace=req.client_id, vector=emb, top_k=3, include_metadata=True, filter={"type": "knowledge"})
        context = "\n".join([m['metadata']['text'] for m in search['matches']])
        
        sys_msg = f"Role: {conf.get('bot_name')}. Language: {conf.get('bot_lang')}. Context: {context}. User: {req.message}"
        ans = get_model(active_key).generate_content(sys_msg).text
        
        log_meta = {"type": "chat_log", "session": req.session_id, "user_msg": req.message, "bot_msg": ans, "timestamp": int(time.time())}
        index.upsert(vectors=[{"id": f"log_{uuid.uuid4()}", "values": [0.1]*768, "metadata": log_meta}], namespace=req.client_id)
        return {"answer": ans}
    except Exception as e: return {"answer": f"System error: {str(e)}"}

@app.post("/get-config")
async def get_conf(req: StatusRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if res.vectors: return res.vectors[f"config_{req.client_id}"].metadata
        return {"bot_name": "Support", "bot_color": "#4F46E5"}
    except: return {"bot_name": "Support", "bot_color": "#4F46E5"}

@app.post("/get-stats")
async def stats_engine(req: StatusRequest):
    chat_res = index.query(namespace=req.client_id, vector=[0.0]*768, filter={"type": "chat_log"}, top_k=10000, include_metadata=True)
    sessions = set([m['metadata'].get('session') for m in chat_res['matches']])
    lead_res = index.query(namespace=req.client_id, vector=[0.0]*768, filter={"type": "lead"}, top_k=10000)
    return {"visitors": len(sessions), "chats": len(chat_res['matches']), "leads": len(lead_res['matches'])}

@app.post("/get-leads")
async def leads_engine(req: StatusRequest):
    res = index.query(namespace=req.client_id, vector=[0.0]*768, filter={"type": "lead"}, top_k=100, include_metadata=True)
    return {"leads": [m['metadata'] for m in res['matches']]}

@app.post("/train")
async def train_saas_engine(req: TrainRequest):
    meta = req.dict(); meta["type"] = "config"
    index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
    return {"status": "success"}

@app.get("/")
def health(): return {"status": "Omni-Brain v27.2 Active"}
