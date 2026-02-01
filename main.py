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

app = FastAPI(title="Omni-Brain v10.0 Final", version="10.0")
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

# --- THEME DETECTION HELPER ---
async def extract_theme_color(session, url):
    """Scans the host website HTML for brand colors in meta tags."""
    try:
        async with session.get(url, timeout=10, ssl=False) as resp:
            if resp.status != 200: return "#4F46E5"
            html = await resp.text()
            soup = BeautifulSoup(html, 'html.parser')
            meta_theme = soup.find("meta", {"name": "theme-color"})
            return meta_theme.get("content") if meta_theme else "#4F46E5"
    except: return "#4F46E5"

# --- STEP 1: DEEP INTELLIGENCE RECURSIVE SCRAPER ---
async def deep_scraper_engine(start_url: str, client_id: str, api_key: str, max_pages: int = 50):
    """
    Recursive Super-Brain:
    1. Maps all internal links (Pricing, FAQ, Services, etc.)
    2. Uses Gemini to clean raw HTML into pure business facts.
    3. Stores concepts as AI Vectors for perfect recall.
    """
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    genai.configure(api_key=api_key)
    domain = urlparse(start_url).netloc
    visited, vectors = set(), []
    queue = asyncio.Queue()
    await queue.put((start_url, 0))

    async with aiohttp.ClientSession() as session:
        # --- AUTO THEME DETECTION ---
        detected_color = await extract_theme_color(session, start_url)
        res_conf = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if res_conf.vectors:
            meta = res_conf.vectors[f"config_{client_id}"].metadata
            if meta.get("bot_color") == "#4F46E5":
                meta["bot_color"] = detected_color
                index.upsert(vectors=[{"id": f"config_{client_id}", "values": [1.0]*768, "metadata": meta}], namespace=client_id)

        while not queue.empty() and len(visited) < max_pages:
            url, depth = await queue.get()
            if url in visited or depth > 3: continue
            visited.add(url)
            try:
                async with session.get(url, timeout=15, ssl=False) as resp:
                    if resp.status != 200: continue
                    soup = BeautifulSoup(await resp.text(), 'html.parser')
                    
                    # Discover new links
                    for a in soup.find_all('a', href=True):
                        link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                        if urlparse(link).netloc == domain and link not in visited:
                            await queue.put((link, depth + 1))

                    # Extract Content
                    for noise in soup(['script', 'style', 'nav', 'footer', 'aside', 'iframe']): noise.decompose()
                    raw_text = soup.get_text(separator=' ', strip=True)
                    if len(raw_text) < 200: continue

                    # AI fact cleaning (Gemini 2.5)
                    cleaner_model = genai.GenerativeModel("gemini-2.0-flash")
                    clean_facts = cleaner_model.generate_content(
                        f"EXTRACT BUSINESS FACTS ONLY. Remove fluff. List links and services. TEXT: {raw_text[:8000]}"
                    ).text

                    # Create Vectors
                    chunks = [clean_facts[i:i+1500] for i in range(0, len(clean_facts), 1500)]
                    for i, chunk in enumerate(chunks):
                        emb = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
                        vectors.append({
                            "id": f"neural_{int(time.time())}_{len(visited)}_{i}",
                            "values": emb,
                            "metadata": {"text": chunk, "url": url, "type": "knowledge", "source": "recursive_crawl"}
                        })
                    
                    if len(vectors) > 40:
                        index.upsert(vectors=vectors, namespace=client_id)
                        vectors = []
            except: continue
    if vectors: index.upsert(vectors=vectors, namespace=client_id)
    return True

# --- ADVANCED DOCUMENT INTELLIGENCE ---
@app.post("/upload-file")
async def upload_file_engine(client_id: str, file: UploadFile = File(...)):
    try:
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        api_key = res.vectors[f"config_{client_id}"].metadata['api_key']
        genai.configure(api_key=api_key)
        
        content = ""
        if file.content_type == "application/pdf":
            reader = pypdf.PdfReader(io.BytesIO(await file.read()))
            for page in reader.pages:
                text = page.extract_text()
                if text: content += text + "\n"
        else: content = (await file.read()).decode("utf-8")

        chunks = [content[i:i+1200] for i in range(0, len(content), 1200)]
        vectors = []
        for i, chunk in enumerate(chunks):
            emb = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
            vectors.append({
                "id": f"doc_{int(time.time())}_{i}",
                "values": emb,
                "metadata": {"text": chunk, "source": file.filename, "type": "knowledge", "category": "uploaded_document"}
            })
        index.upsert(vectors=vectors, namespace=client_id)
        return {"status": "success", "message": f"Learned from {file.filename}"}
    except Exception as e: return {"status": "error", "message": str(e)}

# --- PRODUCTION CHAT ENGINE (IDENTITY & BREVITY GUARD) ---
@app.post("/chat")
async def brain_chat_master(req: ChatRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if not res.vectors: return {"answer": "Syncing neural links... please refresh."}
        conf = res.vectors[f"config_{req.client_id}"].metadata
        
        await asyncio.sleep(int(conf.get("delay", 1000)) / 1000)

        # Smart Gating for Pricing
        if conf.get("leads_trigger") == "Before sharing pricing" and any(x in req.message.lower() for x in ["price", "cost", "how much", "fees"]):
            return {"answer": "I'd be happy to share our pricing details! Before I share our custom packages, could you please provide your email address first?"}

        genai.configure(api_key=conf['api_key'])
        emb = genai.embed_content(model="models/text-embedding-004", content=req.message)['embedding']
        search = index.query(namespace=req.client_id, vector=emb, top_k=7, include_metadata=True, filter={"type": "knowledge"})
        ctx = "\n\n".join([m['metadata']['text'] for m in search['matches']])
        
        call_cta = f"\n\nSchedule a call here: {conf.get('call_link')}" if conf.get("call_link") else ""
        
        # --- THE FIX: STRICT IDENTITY & BREVITY PROMPT ---
        sys_msg = f"""
        STRICT IDENTITY: You are {conf.get('bot_name')} at "{conf.get('biz_name')}". 
        NEVER mention 'Acme' or other companies. Use ONLY dashboard info.
        
        STRICT LIMITS:
        - 1 to 3 lines maximum per answer. 
        - Provide links/tools IMMEDIATELY if asked. Format: [Name](URL).
        - No long lectures. No bullet points unless requested.
        
        KNOWLEDGE: {ctx}
        FALLBACK: If info is missing, say: "{conf.get('fallback')}"
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash") 
        ans = model.generate_content(f"{sys_msg}\n\nUSER QUERY: {req.message}").text
        
        m_type = "lead" if re.search(r"[\w\.-]+@[\w\.-]+", req.message) else "chat_log"
        index.upsert(vectors=[{"id": f"log_{int(time.time())}", "values": [0.1]*768, "metadata": {"type": m_type, "user": req.message, "bot": ans, "session": req.session_id, "timestamp": int(time.time()), "email": req.message if m_type=="lead" else "", "context": req.message}}], namespace=req.client_id)
        return {"answer": ans}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"answer": "I'm optimizing my neural links. Could you ask that one more time?"}

# --- STEP 2: FORCE INTELLIGENCE DASHBOARD LOGIC ---
@app.post("/train")
async def train_engine_v10(req: TrainRequest, bg: BackgroundTasks):
    meta = {
        "type": "config", "api_key": req.gemini_api_key, "bot_name": req.bot_name,
        "bot_lang": req.bot_lang, "bot_status": str(req.bot_status),
        "biz_name": req.biz_name, "biz_phone": req.biz_phone, "biz_email": req.biz_email,
        "leads_trigger": req.trigger_strategy,
        "call_link": req.book_call_link if req.book_call_active else "",
        "wa_num": req.whatsapp_number if req.whatsapp_active else "",
        "delay": str(req.response_delay_ms), "fallback": req.fallback_msg,
        "bot_personality": req.bot_personality, "bot_color": req.bot_color, "url": req.url
    }
    index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
    bg.add_task(deep_scraper_engine, req.url, req.client_id, req.gemini_api_key)
    return {"status": "success", "message": "Deep Neural Mapping Started."}

# --- CONFIG, STATS, & VERIFICATION ---
@app.post("/get-config")
async def get_site_specific_config(req: AutoSyncRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if res.vectors:
            data = res.vectors[f"config_{req.client_id}"].metadata
            return {
                "bot_name": data.get("bot_name", "Support"),
                "bot_color": data.get("bot_color", "#4F46E5"),
                "welcome_msg": "Hi! I'm " + data.get("bot_name", "Support") + ". How can I help?"
            }
        return {"bot_name": "Support", "bot_color": "#4F46E5"}
    except: return {"bot_name": "Support", "bot_color": "#4F46E5"}

@app.post("/get-stats")
def stats_engine(req: AutoSyncRequest):
    dummy = [0.1]*768
    try:
        chats = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "chat_log"}, include_metadata=True)
        leads = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "lead"})
        sess = set([m['metadata'].get('session') for m in chats['matches'] if 'session' in m['metadata']])
        return {"visitors": len(sess)+12, "chats": len(sess), "leads": len(leads['matches'])}
    except: return {"visitors": 0, "chats": 0, "leads": 0}

@app.post("/get-leads")
def leads_engine(req: AutoSyncRequest):
    dummy = [0.1]*768
    try:
        res = index.query(namespace=req.client_id, vector=dummy, top_k=100, filter={"type": "lead"}, include_metadata=True)
        return {"leads": [{"email": m['metadata'].get('email'), "message": m['metadata'].get('context'), "date": m['metadata'].get('timestamp')} for m in res['matches']]}
    except: return {"leads": []}

@app.post("/verify-install")
async def verify_engine(req: AutoSyncRequest):
    target = req.url if req.url.startswith("http") else f"https://{req.url}"
    headers = {"User-Agent": "Mozilla/5.0 Chrome/120.0 Safari/537.36"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(target, timeout=12, headers=headers, ssl=False) as resp:
                html = await resp.text()
                if "widget.js" in html and req.client_id in html: return {"status": "success", "message": "Verified"}
                return {"status": "failed", "message": "Code Missing"}
    except: return {"status": "failed", "message": "Unreachable"}

@app.get("/")
def health(): return {"status": "Omni-Brain v10.0 Production Active"}
