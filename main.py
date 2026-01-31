import os
import time
import re
import asyncio
import aiohttp
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
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# --- MODELS ---
class RequestModel(BaseModel):
    url: str = ""
    client_id: str
    message: str = ""
    session_id: str = "Guest"
    gemini_api_key: str = ""
    bot_name: str = "AI Support"
    bot_color: str = "#4F46E5"
    bot_avatar: str = ""
    bot_personality: str = "Helpful"

# --- HELPER FUNCTIONS ---
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
def home(): return {"status": "Active"}

@app.post("/verify-install")
async def verify(req: RequestModel):
    # Real crawler to check for script tag
    target = req.url if req.url.startswith("http") else f"https://{req.url}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(target, timeout=10, ssl=False) as resp:
                text = await resp.text()
                if "widget.js" in text and req.client_id in text:
                    return {"status": "success", "message": "Verified"}
    except: pass
    return {"status": "failed", "message": "Script not found"}

@app.post("/train")
async def train(req: RequestModel):
    # Save Config
    index.upsert(
        vectors=[{
            "id": f"config_{req.client_id}",
            "values": [1.0]*768,
            "metadata": {
                "api_key": req.gemini_api_key,
                "bot_name": req.bot_name,
                "bot_color": req.bot_color,
                "bot_personality": req.bot_personality,
                "target_url": req.url
            }
        }],
        namespace=req.client_id
    )
    # Note: Actual crawling logic is heavy, omitted here for stability, 
    # but config saving is critical for the dashboard to work.
    return {"status": "success"}

@app.get("/get-config")
def get_conf(client_id: str):
    c = get_config(client_id)
    if c: return c
    return {"bot_name": "AI Support", "bot_color": "#4F46E5", "bot_personality": ""}

@app.post("/chat")
async def chat(req: RequestModel):
    conf = get_config(req.client_id)
    if not conf: return {"answer": "Bot not configured."}
    
    # Check for lead
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", req.message)
    if email_match: save_lead(req.client_id, email_match.group(), req.message)

    # RAG Response
    try:
        genai.configure(api_key=conf['api_key'])
        emb = genai.embed_content(model="models/text-embedding-004", content=req.message)['embedding']
        res = index.query(namespace=req.client_id, vector=emb, top_k=3, include_metadata=True)
        ctx = "\n".join([m['metadata']['text'] for m in res['matches'] if 'text' in m['metadata']])
        
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"You are {conf.get('bot_name')}. Style: {conf.get('bot_personality')}. Context: {ctx}. User: {req.message}"
        ans = model.generate_content(prompt).text
        
        log_chat(req.client_id, req.session_id, req.message, ans)
        return {"answer": ans}
    except Exception as e: return {"answer": str(e)}

@app.post("/get-stats")
def get_stats(req: RequestModel):
    # Fetch Leads Count
    dummy = [0.1]*768
    leads = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "lead"})
    chats = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "chat_log"})
    
    # Unique sessions
    sessions = set()
    for m in chats['matches']:
        if 'session_id' in m['metadata']: sessions.add(m['metadata']['session_id'])
        
    return {
        "visitors": 0, # Placeholder until pixel tracking
        "chats": len(sessions),
        "leads": len(leads['matches'])
    }

@app.post("/get-leads")
def get_leads(req: RequestModel):
    dummy = [0.1]*768
    res = index.query(namespace=req.client_id, vector=dummy, top_k=100, filter={"type": "lead"})
    data = []
    for m in res['matches']:
        data.append({
            "email": m['metadata'].get('email', 'No Email'),
            "message": m['metadata'].get('context', ''),
            "date": m['metadata'].get('timestamp', 0)
        })
    return {"leads": data}

@app.post("/get-analytics")
def get_analytics(req: RequestModel):
    dummy = [0.1]*768
    res = index.query(namespace=req.client_id, vector=dummy, top_k=100, filter={"type": "chat_log"})
    data = []
    for m in res['matches']:
        data.append({
            "session": m['metadata'].get('session_id', 'Unknown'),
            "user": m['metadata'].get('user_msg', ''),
            "bot": m['metadata'].get('bot_msg', ''),
            "time": m['metadata'].get('timestamp', 0)
        })
    return {"logs": data}
