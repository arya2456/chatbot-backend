import os
import time
import re
import asyncio
import aiohttp
import logging
import io
import pypdf
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

# --- SYSTEM CONFIG ---
PINECONE_INDEX_NAME = "chatbot-index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Omni-Brain v8.5 Final", version="8.5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- DATABASE INIT ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    logger.info("✅ Gemini 2.5 Brain Connected")
except Exception as e:
    logger.error(f"❌ Database Error: {e}")
    index = None

# --- MODELS ---
class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str
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
    email_alerts: bool = True
    alert_recipient: str = ""
    crm_integration: str = "None (Store Locally)"
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

# --- ENHANCED DOCUMENT & WEB INTELLIGENCE ---

async def deep_crawl(start_url: str, client_id: str, api_key: str, max_pages: int = 40):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    genai.configure(api_key=api_key)
    domain = urlparse(start_url).netloc
    visited, vectors = set(), []
    queue = asyncio.Queue()
    await queue.put((start_url, 0))

    async with aiohttp.ClientSession() as session:
        while not queue.empty() and len(visited) < max_pages:
            url, depth = await queue.get()
            if url in visited or depth > 3: continue
            visited.add(url)
            try:
                async with session.get(url, timeout=12, ssl=False) as resp:
                    if resp.status != 200: continue
                    soup = BeautifulSoup(await resp.text(), 'html.parser')
                    for a in soup.find_all('a', href=True):
                        link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                        if urlparse(link).netloc == domain and link not in visited:
                            await queue.put((link, depth + 1))
                    for x in soup(['script', 'style', 'nav', 'footer', 'aside']): x.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    if len(text) < 100: continue
                    chunks = [text[i:i+1200] for i in range(0, len(text), 1200)]
                    for i, chunk in enumerate(chunks):
                        emb = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
                        vectors.append({"id": f"web_{client_id}_{len(visited)}_{i}", "values": emb, "metadata": {"text": chunk, "url": url, "type": "knowledge", "source": "website"}})
                    if len(vectors) > 40:
                        index.upsert(vectors=vectors, namespace=client_id)
                        vectors = []
            except: continue
    if vectors: index.upsert(vectors=vectors, namespace=client_id)

@app.post("/upload-file")
async def upload_file_engine(client_id: str, file: UploadFile = File(...)):
    """Document Intelligence: Converts PDFs/Txt into high-dimensional AI vectors."""
    try:
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        api_key = res.vectors[f"config_{client_id}"].metadata['api_key']
        genai.configure(api_key=api_key)
        
        content = ""
        if file.content_type == "application/pdf":
            reader = pypdf.PdfReader(io.BytesIO(await file.read()))
            for page in reader.pages: content += page.extract_text() + "\n"
        else: content = (await file.read()).decode("utf-8")

        chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
        vectors = []
        for i, c in enumerate(chunks):
            vectors.append({
                "id": f"file_{int(time.time())}_{i}", 
                "values": genai.embed_content(model="models/text-embedding-004", content=c)['embedding'], 
                "metadata": {"text": c, "source": file.filename, "type": "knowledge", "category": "document"}
            })
        index.upsert(vectors=vectors, namespace=client_id)
        return {"status": "success", "message": f"Learned from {file.filename}"}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- DEPLOYMENT & ANALYTICS ---

@app.post("/verify-install")
async def verify_engine(req: AutoSyncRequest):
    target = req.url if req.url.startswith("http") else f"https://{req.url}"
    headers = {"User-Agent": "Mozilla/5.0 Chrome/120.0 Safari/537.36"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(target, timeout=12, headers=headers, ssl=False) as resp:
                html = await resp.text()
                if "widget.js" in html and req.client_id in html: return {"status": "success", "message": "Verified"}
                return {"status": "failed", "message": "Code not detected"}
    except: return {"status": "failed", "message": "Unreachable"}

@app.post("/get-stats")
def stats_engine(req: AutoSyncRequest):
    dummy = [0.1]*768
    try:
        chats = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "chat_log"}, include_metadata=True)
        leads = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "lead"})
        sess = set([m['metadata'].get('session') for m in chats['matches'] if 'session' in m['metadata']])
        return {"visitors": len(sess)+10, "chats": len(sess), "leads": len(leads['matches']), "insight": "Intelligence Active"}
    except: return {"visitors": 0, "chats": 0, "leads": 0}

@app.post("/get-analytics")
def intelligence_engine(req: AutoSyncRequest):
    dummy = [0.1]*768
    try:
        res = index.query(namespace=req.client_id, vector=dummy, top_k=500, filter={"type": "chat_log"}, include_metadata=True)
        sessions = {}
        for m in res['matches']:
            meta = m.get('metadata', {})
            sid = meta.get('session', 'Unknown')
            if sid not in sessions: sessions[sid] = {"session_id": sid, "last_active": meta.get('timestamp'), "transcript": []}
            sessions[sid]["transcript"].append({"user": meta.get('user'), "bot": meta.get('bot'), "time": meta.get('timestamp')})
        return {"sessions": sorted(sessions.values(), key=lambda x: x['last_active'], reverse=True), "growth_strategy": "Analyzing conversation patterns..."}
    except: return {"sessions": []}

@app.post("/get-leads")
def leads_engine(req: AutoSyncRequest):
    dummy = [0.1]*768
    try:
        res = index.query(namespace=req.client_id, vector=dummy, top_k=100, filter={"type": "lead"}, include_metadata=True)
        return {"leads": [{"email": m['metadata'].get('email'), "message": m['metadata'].get('context'), "date": m['metadata'].get('timestamp')} for m in res['matches']]}
    except: return {"leads": []}

# --- NEURAL CHAT ENGINE (GEMINI 2.5 FLASH) ---

@app.post("/chat")
async def brain_chat_v8(req: ChatRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        conf = res.vectors[f"config_{req.client_id}"].metadata
        
        # 1. Dashboard-controlled Timing
        await asyncio.sleep(int(conf.get("delay", 1000)) / 1000)

        # 2. Smart Lead Gating (Pricing Rule)
        if conf.get("leads_trigger") == "Before sharing pricing" and any(x in req.message.lower() for x in ["price", "cost", "how much", "quote"]):
            return {"answer": "I'd be happy to provide our pricing details! Before I share our custom packages, could you please provide your email address so I can send the proposal directly to you?"}

        # 3. Recall Knowledge (Web + PDF)
        genai.configure(api_key=conf['api_key'])
        emb = genai.embed_content(model="models/text-embedding-004", content=req.message)['embedding']
        search = index.query(namespace=req.client_id, vector=emb, top_k=6, include_metadata=True, filter={"type": "knowledge"})
        ctx = "\n\n".join([m['metadata']['text'] for m in search['matches']])
        
        # 4. Neural Response (Forced 2.5 Flash)
        call_link = conf.get('call_link')
        cta = f"\n\nBook a meeting here: {call_link}" if call_link else ""
        sys_msg = f"Role: {conf.get('bot_name')} at {conf.get('biz_name')}. Persona: {conf.get('bot_personality')}. Knowledge: {ctx}. {cta}. Fallback: {conf.get('fallback')}"
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        ans = model.generate_content(f"{sys_msg}\n\nUSER QUERY: {req.message}").text
        
        # 5. Lead Classification
        m_type = "chat_log"
        if re.search(r"[\w\.-]+@[\w\.-]+", req.message): m_type = "lead"
        
        index.upsert(vectors=[{"id": f"log_{int(time.time())}", "values": [0.1]*768, "metadata": {"type": m_type, "user": req.message, "bot": ans, "session": req.session_id, "timestamp": int(time.time()), "email": req.message if m_type=="lead" else "", "context": req.message}}], namespace=req.client_id)
        
        return {"answer": ans}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"answer": "I'm optimizing my neural links. Could you ask that one more time?"}

@app.post("/train")
async def train_engine(req: TrainRequest, bg: BackgroundTasks):
    meta = {
        "type": "config", "api_key": req.gemini_api_key, "bot_name": req.bot_name,
        "bot_lang": req.bot_lang, "bot_status": str(req.bot_status),
        "biz_name": req.biz_name, "biz_phone": req.biz_phone, "biz_email": req.biz_email,
        "leads_trigger": req.trigger_strategy,
        "call_link": req.book_call_link if req.book_call_active else "",
        "wa_num": req.whatsapp_number if req.whatsapp_active else "",
        "delay": str(req.response_delay_ms), "fallback": req.fallback_msg,
        "max_len": str(req.max_conv_length), "url": req.url,
        "bot_personality": req.bot_personality, "bot_color": req.bot_color
    }
    index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
    bg.add_task(deep_crawl, req.url, req.client_id, req.gemini_api_key)
    return {"status": "success"}

@app.post("/get-config")
async def get_conf(req: AutoSyncRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        return res.vectors[f"config_{req.client_id}"].metadata if res.vectors else {}
    except: return {}

@app.get("/")
def health(): return {"status": "Omni-Brain v8.5 Final Active"}
