import os
import time
import re
import asyncio
import aiohttp
import colorsys
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# --- SYSTEM CONFIG ---
PINECONE_INDEX_NAME = "chatbot-index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Omni-Brain v6.0", version="6.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- DATABASE INIT ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except: index = None

# --- MODELS ---
class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str
    bot_name: str = "AI Assistant"
    bot_color: str = "#4F46E5"
    bot_personality: str = "Expert Consultant"
    auto_theme: bool = True
    bot_status: bool = True
    bot_lang: str = "English"
    biz_name: str = ""
    biz_phone: str = ""
    biz_email: str = ""
    fallback_msg: str = "I'd love to help with that, but I need a human to confirm. Can I have your email?"

class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"
    page_url: str = ""

class AutoSyncRequest(BaseModel):
    client_id: str
    url: str = ""

# --- SMART CRAWLER ---
async def crawl_and_index(start_url: str, client_id: str, api_key: str):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    genai.configure(api_key=api_key)
    domain = urlparse(start_url).netloc
    visited, vectors = set(), []
    
    async with aiohttp.ClientSession() as session:
        queue = [start_url]
        while queue and len(visited) < 25: # Expanded crawl depth
            url = queue.pop(0)
            if url in visited: continue
            visited.add(url)
            try:
                async with session.get(url, timeout=12, ssl=False) as resp:
                    if resp.status != 200: continue
                    soup = BeautifulSoup(await resp.text(), 'html.parser')
                    
                    # 1. Extract Vital Links (Audit tools, Contact, Pricing)
                    nav_links = []
                    for a in soup.find_all('a', href=True):
                        link = urljoin(url, a['href'])
                        text = a.get_text(strip=True)
                        if text and "http" in link:
                            nav_links.append(f"Button/Link: {text} -> {link}")
                        if urlparse(link).netloc == domain and link not in visited:
                            queue.append(link)
                    
                    # 2. Extract Text
                    for x in soup(['script', 'style', 'nav', 'footer']): x.decompose()
                    content = soup.get_text(separator=' ', strip=True)
                    
                    # 3. Create Multi-Dimensional Context
                    full_payload = f"URL: {url}\nTEXT: {content}\nLINKS: {' | '.join(nav_links[:15])}"
                    
                    # 4. Neural Chunking
                    chunks = [full_payload[i:i+1200] for i in range(0, len(full_payload), 1200)]
                    for i, chunk in enumerate(chunks):
                        emb = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
                        vectors.append({
                            "id": f"{client_id}_{len(visited)}_{i}",
                            "values": emb,
                            "metadata": {"text": chunk, "url": url, "type": "knowledge"}
                        })
            except: continue
            
    if vectors: 
        # Batch upsert to prevent timeout
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i:i+100], namespace=client_id)

# --- THE BRAIN CORE ---
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # 1. Identity Verification
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if not res.vectors: return {"answer": "I'm still learning about your business. Please train me in the dashboard."}
        conf = res.vectors[f"config_{req.client_id}"].metadata
        
        if conf.get("bot_status") == "False": return {"answer": "System maintenance in progress."}

        # 2. Knowledge Retrieval (Deep Search)
        genai.configure(api_key=conf['api_key'])
        emb = genai.embed_content(model="models/text-embedding-004", content=req.message)['embedding']
        search = index.query(namespace=req.client_id, vector=emb, top_k=6, include_metadata=True, filter={"type": "knowledge"})
        context = "\n\n---\n\n".join([m['metadata']['text'] for m in search['matches']])
        
        # 3. Intelligence Logic
        sys_prompt = f"""
        Identity: You are {conf.get('bot_name')}, a high-level executive at {conf.get('biz_name')}.
        Personality: {conf.get('bot_personality')}.
        Tone: Professional, expert, helpful.
        Language: {conf.get('bot_lang')}.

        KNOWLEDGE BASE:
        {context}

        INSTRUCTIONS:
        1. If the user asks for a link, tool, or audit, SEARCH the 'LINKS' section of the context and provide the EXACT URL.
        2. Format all links like this: [Click Here](URL).
        3. If the user is asking about services, emphasize benefits.
        4. If the answer is absolutely not in the Knowledge Base, use this exact fallback: "{conf.get('fallback_msg')}"
        5. NEVER mention you are an AI or that you have a "Knowledge Base". Speak as a human employee.
        """
        
        # 4. Model Selection (Gemini 2.5 Flash with 1.5 Auto-Fallback)
        try:
            model = genai.GenerativeModel("gemini-1.5-flash") # Stable for production
            ans = model.generate_content(f"{sys_prompt}\n\nUSER QUERY: {req.message}").text
        except:
            return {"answer": conf.get('fallback_msg')}

        # 5. CRM & Lead Capture (Email detection)
        if re.search(r"[\w\.-]+@[\w\.-]+", req.message):
            lid = f"lead_{int(time.time())}"
            index.upsert(vectors=[{"id": lid, "values": [0.1]*768, "metadata": {"type": "lead", "email": req.message, "context": req.message, "timestamp": int(time.time())}}], namespace=req.client_id)

        # 6. Interaction Logging
        log_id = f"log_{int(time.time())}_{abs(hash(req.message))}"
        index.upsert(vectors=[{"id": log_id, "values": [0.1]*768, "metadata": {"type": "chat_log", "user": req.message, "bot": ans, "session": req.session_id, "timestamp": int(time.time())}}], namespace=req.client_id)
        
        return {"answer": ans}
        
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return {"answer": "I'm experiencing a brief neural glitch. Could you try that again?"}

# --- OTHER ENDPOINTS (Standardized) ---
@app.get("/")
def home(): return {"status": "Omni-Brain v6.0 Active"}

@app.post("/train")
async def train(req: TrainRequest, bg: BackgroundTasks):
    meta = {
        "type": "config", "api_key": req.gemini_api_key, "bot_name": req.bot_name,
        "bot_personality": req.bot_personality, "bot_color": req.bot_color, "target_url": req.url,
        "bot_status": str(req.bot_status), "bot_lang": req.bot_lang, "biz_name": req.biz_name,
        "biz_phone": req.biz_phone, "biz_email": req.biz_email, "fallback_msg": req.fallback_msg
    }
    index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
    bg.add_task(crawl_and_index, req.url, req.client_id, req.gemini_api_key)
    return {"status": "success"}

@app.post("/get-stats")
def stats(req: AutoSyncRequest):
    dummy = [0.1]*768
    chats = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "chat_log"})
    leads = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "lead"})
    sess = set(m['metadata']['session'] for m in chats['matches'] if 'session' in m['metadata'])
    return {"visitors": 0, "chats": len(sess), "leads": len(leads['matches'])}

@app.post("/get-leads")
def leads(req: AutoSyncRequest):
    dummy = [0.1]*768
    res = index.query(namespace=req.client_id, vector=dummy, top_k=100, filter={"type": "lead"})
    return {"leads": [{"email": m['metadata'].get('email'), "message": m['metadata'].get('context'), "date": m['metadata'].get('timestamp')} for m in res['matches']]}

@app.post("/get-analytics")
def analytics(req: AutoSyncRequest):
    dummy = [0.1]*768
    res = index.query(namespace=req.client_id, vector=dummy, top_k=100, filter={"type": "chat_log"})
    return {"logs": [{"session": m['metadata'].get('session'), "user": m['metadata'].get('user'), "bot": m['metadata'].get('bot'), "time": m['metadata'].get('timestamp')} for m in res['matches']]}

@app.post("/get-config")
async def get_conf(req: AutoSyncRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        return res.vectors[f"config_{req.client_id}"].metadata if res.vectors else {}
    except: return {}
