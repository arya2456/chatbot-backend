import os
import time
import re
import asyncio
import aiohttp
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

# --- CONFIG ---
PINECONE_INDEX_NAME = "chatbot-index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB INIT ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except: pass

# --- MODELS ---
class SettingsModel(BaseModel):
    client_id: str
    url: str = ""
    gemini_api_key: str = ""
    # General
    bot_name: str = "AI Support"
    bot_status: bool = True
    bot_lang: str = "English"
    # Business
    biz_name: str = ""
    biz_phone: str = ""
    biz_email: str = ""
    # Behavior
    bot_personality: str = "Professional"
    fallback_msg: str = "I am not sure about that. Would you like to speak to a human?"
    # Leads
    lead_trigger: str = "before_pricing"

class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"

class AutoSyncRequest(BaseModel):
    client_id: str
    url: str = ""

# --- HELPERS ---
def get_config(client_id):
    try:
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if f"config_{client_id}" in res.vectors:
            return res.vectors[f"config_{client_id}"].metadata
        return None
    except: return None

def save_lead(client_id, email, context):
    try:
        lid = f"lead_{int(time.time())}_{abs(hash(email))}"
        index.upsert(
            vectors=[{"id": lid, "values": [0.1]*768, "metadata": {"type": "lead", "email": email, "context": context, "timestamp": int(time.time())}}],
            namespace=client_id
        )
    except: pass

def log_chat(client_id, session, user, bot):
    try:
        lid = f"log_{int(time.time())}_{abs(hash(user))}"
        index.upsert(
            vectors=[{"id": lid, "values": [0.1]*768, "metadata": {"type": "chat_log", "session_id": session, "user_msg": user, "bot_msg": bot, "timestamp": int(time.time())}}],
            namespace=client_id
        )
    except: pass

# --- ENDPOINTS ---

@app.get("/")
def home(): return {"status": "FC Brain v2.5 Online"}

@app.post("/verify-install")
async def verify(req: AutoSyncRequest):
    target = req.url if req.url.startswith("http") else f"https://{req.url}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(target, timeout=10, ssl=False) as resp:
                text = await resp.text()
                if "widget.js" in text and req.client_id in text:
                    return {"status": "success", "message": "Widget Detected & Active"}
    except: pass
    return {"status": "failed", "message": "Script not found on homepage"}

@app.post("/train")
async def save_settings(req: SettingsModel):
    # Upsert Configuration
    meta = {
        "type": "config",
        "api_key": req.gemini_api_key,
        "bot_name": req.bot_name,
        "bot_status": str(req.bot_status),
        "bot_lang": req.bot_lang,
        "biz_name": req.biz_name,
        "biz_phone": req.biz_phone,
        "biz_email": req.biz_email,
        "bot_personality": req.bot_personality,
        "fallback_msg": req.fallback_msg,
        "target_url": req.url
    }
    
    try:
        index.upsert(
            vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}],
            namespace=req.client_id
        )
        return {"status": "success", "message": "Settings Saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/get-config")
def get_conf(client_id: str):
    c = get_config(client_id)
    if c: return c
    return {"bot_name": "AI Support", "bot_color": "#4F46E5"}

@app.post("/chat")
async def chat(req: ChatRequest):
    conf = get_config(req.client_id)
    if not conf: return {"answer": "Bot not configured."}
    
    # Check Paused Status
    if conf.get("bot_status") == "False":
        return {"answer": "I am currently offline. Please contact support directly."}

    # Save Leads
    if re.search(r"[\w\.-]+@[\w\.-]+", req.message): 
        save_lead(req.client_id, req.message, req.message)

    try:
        genai.configure(api_key=conf.get('api_key'))
        
        # 1. Search Knowledge
        emb = genai.embed_content(model="models/text-embedding-004", content=req.message)['embedding']
        res = index.query(namespace=req.client_id, vector=emb, top_k=3, include_metadata=True)
        knowledge = "\n".join([m['metadata']['text'] for m in res['matches'] if 'text' in m['metadata']])
        
        # 2. Build Smart System Prompt
        system = f"""
        You are {conf.get('bot_name', 'AI Assistant')}, representing {conf.get('biz_name', 'the company')}.
        Personality: {conf.get('bot_personality', 'Professional')}.
        Language: {conf.get('bot_lang', 'English')}.
        
        BUSINESS DETAILS:
        - Support Email: {conf.get('biz_email', 'N/A')}
        - Phone: {conf.get('biz_phone', 'N/A')}
        
        KNOWLEDGE BASE:
        {knowledge}
        
        RULES:
        1. Answer strictly based on the Knowledge Base.
        2. If the answer is NOT in the knowledge base, say exactly: "{conf.get('fallback_msg')}"
        3. Be concise and helpful.
        """
        
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        ans = model.generate_content(f"{system}\n\nUser: {req.message}").text
        
        log_chat(req.client_id, req.session_id, req.message, ans)
        return {"answer": ans}
        
    except: return {"answer": conf.get('fallback_msg')}

@app.post("/get-stats")
def get_stats(req: AutoSyncRequest):
    dummy = [0.1]*768
    leads = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "lead"})
    chats = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "chat_log"})
    sessions = set([m['metadata'].get('session_id') for m in chats['matches']])
    return {"visitors": 0, "chats": len(sessions), "leads": len(leads['matches'])}

@app.post("/get-leads")
def get_leads_ep(req: AutoSyncRequest):
    dummy = [0.1]*768
    res = index.query(namespace=req.client_id, vector=dummy, top_k=100, filter={"type": "lead"})
    data = [{"email": m['metadata'].get('email'), "message": m['metadata'].get('context'), "date": m['metadata'].get('timestamp')} for m in res['matches']]
    return {"leads": data}

@app.post("/get-analytics")
def get_analytics_ep(req: AutoSyncRequest):
    dummy = [0.1]*768
    res = index.query(namespace=req.client_id, vector=dummy, top_k=100, filter={"type": "chat_log"})
    data = [{"session": m['metadata'].get('session_id'), "user": m['metadata'].get('user_msg'), "bot": m['metadata'].get('bot_msg'), "time": m['metadata'].get('timestamp')} for m in res['matches']]
    return {"logs": data}
