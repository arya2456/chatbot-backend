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

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = "chatbot-index"
# SECURE: Reads from Server Environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FC Media Adaptive Brain", version="4.6")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE INIT ---
try:
    if not PINECONE_API_KEY:
        logger.warning("⚠️ PINECONE_API_KEY not found in environment variables.")
    else:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        logger.info("✅ Connected to Pinecone successfully")
except Exception as e:
    logger.critical(f"❌ Failed to connect to Pinecone: {e}")
    index = None

# --- SCHEDULER ---
scheduler = AsyncIOScheduler()
scheduler.start()

# --- MODELS ---
class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str
    bot_name: str = "AI Assistant"
    bot_color: str = "#4F46E5"
    bot_personality: str = "Professional"
    auto_theme: bool = True
    bot_status: bool = True
    bot_lang: str = "English"
    biz_name: str = ""
    biz_phone: str = ""
    biz_email: str = ""
    fallback_msg: str = "I am not sure. Would you like to speak to a human?"

class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"
    page_url: str = ""

class AutoSyncRequest(BaseModel):
    client_id: str
    url: str = ""

# --- THEME ENGINE ---
async def extract_website_theme(url: str):
    if not url.startswith("http"): url = f"https://{url}"
    theme = {
        "primary_color": "#4F46E5",
        "secondary_color": "#10B981",
        "font_family": "Inter, sans-serif",
        "bg_color": "#FFFFFF",
        "text_color": "#1F2937",
        "border_radius": "12px"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10, ssl=False) as resp:
                if resp.status != 200: return theme
                text = await resp.text()
                soup = BeautifulSoup(text, 'html.parser')
                
                meta = soup.find('meta', {'name': 'theme-color'})
                if meta and meta.get('content'):
                    theme['primary_color'] = meta['content']
                
                styles = "".join([s.string or "" for s in soup.find_all('style')])
                p_match = re.search(r'--primary[^:]*:\s*([#\w]+)', styles)
                if p_match: theme['primary_color'] = p_match.group(1)
                
                f_match = re.search(r'font-family:\s*([^;]+)', styles)
                if f_match: theme['font_family'] = f_match.group(1).split(',')[0].strip().replace('"', '').replace("'", "")

                if theme['primary_color'].startswith('#'):
                    try:
                        h = theme['primary_color'].lstrip('#')
                        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                        hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
                        rgb2 = colorsys.hsv_to_rgb(hsv[0], hsv[1], max(0, hsv[2]-0.2))
                        theme['secondary_color'] = '#%02x%02x%02x' % (int(rgb2[0]*255), int(rgb2[1]*255), int(rgb2[2]*255))
                    except: pass

    except Exception as e:
        logger.error(f"Theme Error: {e}")
    
    return theme

# --- CRAWLER & INDEXER ---
def smart_chunk(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

async def crawl_and_index(start_url: str, client_id: str, api_key: str):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    
    genai.configure(api_key=api_key)
    domain = urlparse(start_url).netloc
    visited = set()
    vectors = []
    
    try:
        async with aiohttp.ClientSession() as session:
            queue = [start_url]
            crawled_count = 0
            
            while queue and crawled_count < 15:
                url = queue.pop(0)
                if url in visited: continue
                visited.add(url)
                
                try:
                    async with session.get(url, timeout=10, ssl=False) as resp:
                        if resp.status != 200: continue
                        html = await resp.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        for x in soup(['script', 'style', 'nav', 'footer', 'svg']): x.decompose()
                        
                        title = soup.title.string if soup.title else ""
                        text = soup.get_text(separator=' ', strip=True)
                        if len(text) < 100: continue
                        
                        full_content = f"URL: {url}\nTITLE: {title}\nCONTENT: {text}"
                        
                        for a in soup.find_all('a', href=True):
                            link = urljoin(url, a['href'])
                            if urlparse(link).netloc == domain and link not in visited:
                                queue.append(link)
                        
                        chunks = smart_chunk(full_content)
                        for i, chunk in enumerate(chunks):
                            emb = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
                            vectors.append({
                                "id": f"{client_id}_{crawled_count}_{i}",
                                "values": emb,
                                "metadata": {"text": chunk, "url": url, "type": "knowledge"}
                            })
                        
                        crawled_count += 1
                        
                except: pass
                
        if vectors:
            index.upsert(vectors=vectors, namespace=client_id)
            return True
            
    except Exception as e:
        logger.error(f"Crawl Failed: {e}")
        return False
    return True

# --- API ENDPOINTS ---

@app.get("/")
def home(): return {"status": "FC Brain 4.6 Active"}

@app.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    theme = {}
    if req.auto_theme:
        theme = await extract_website_theme(req.url)
    
    config_meta = {
        "type": "config",
        "api_key": req.gemini_api_key,
        "bot_name": req.bot_name,
        "bot_personality": req.bot_personality,
        "bot_color": req.bot_color,
        "target_url": req.url,
        "bot_status": str(req.bot_status),
        "bot_lang": req.bot_lang,
        "biz_name": req.biz_name,
        "biz_phone": req.biz_phone,
        "biz_email": req.biz_email,
        "fallback_msg": req.fallback_msg,
        **theme
    }
    
    try:
        index.upsert(
            vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": config_meta}],
            namespace=req.client_id
        )
    except: return {"status": "error", "message": "Database connection failed"}

    background_tasks.add_task(crawl_and_index, req.url, req.client_id, req.gemini_api_key)
    return {"status": "success", "message": "Training started", "theme_detected": theme}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if not res.vectors: return {"answer": "Bot not configured."}
        conf = res.vectors[f"config_{req.client_id}"].metadata
    except: return {"answer": "Database Error."}

    if conf.get("bot_status") == "False":
        return {"answer": "I am currently offline. Please check back later."}

    genai.configure(api_key=conf['api_key'])
    try:
        emb = genai.embed_content(model="models/text-embedding-004", content=req.message)['embedding']
        search = index.query(namespace=req.client_id, vector=emb, top_k=4, include_metadata=True, filter={"type": "knowledge"})
        context = "\n\n".join([m['metadata']['text'] for m in search['matches']])
        
        sys_prompt = f"""
        You are {conf.get('bot_name', 'AI')}. 
        Personality: {conf.get('bot_personality', 'Helpful')}.
        Language: {conf.get('bot_lang', 'English')}.
        
        BUSINESS INFO:
        - Name: {conf.get('biz_name')}
        - Email: {conf.get('biz_email')}
        - Phone: {conf.get('biz_phone')}
        
        CONTEXT FROM WEBSITE:
        {context}
        
        USER QUERY: {req.message}
        CURRENT PAGE: {req.page_url}
        
        INSTRUCTIONS:
        - Answer strictly based on the provided context.
        - If the answer is not in the context, say exactly: "{conf.get('fallback_msg')}"
        - Keep answers concise (under 3 sentences) unless asked for details.
        - Format links as markdown [Link Text](URL).
        """
        
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        ans = model.generate_content(sys_prompt).text
        
        lid = f"log_{int(time.time())}_{abs(hash(req.message))}"
        index.upsert(
            vectors=[{"id": lid, "values": [0.1]*768, "metadata": {"type": "chat_log", "user": req.message, "bot": ans, "session": req.session_id, "timestamp": int(time.time())}}],
            namespace=req.client_id
        )
        
        return {"answer": ans}
        
    except Exception as e:
        return {"answer": "I'm having trouble thinking right now. Please try again."}

@app.post("/get-config")
async def get_configuration(req: AutoSyncRequest):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        if res.vectors: return res.vectors[f"config_{req.client_id}"].metadata
    except: pass
    return {"bot_name": "AI", "primary_color": "#4F46E5"}

@app.post("/verify-install")
async def verify(req: AutoSyncRequest):
    target = req.url if req.url.startswith("http") else f"https://{req.url}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(target, timeout=10, ssl=False) as resp:
                text = await resp.text()
                if "widget.js" in text and req.client_id in text:
                    return {"status": "success", "message": "Widget Detected"}
    except: pass
    return {"status": "failed", "message": "Widget Not Found"}

@app.post("/get-stats")
def stats(req: AutoSyncRequest):
    dummy = [0.1]*768
    chats = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "chat_log"})
    sessions = set(m['metadata']['session'] for m in chats['matches'] if 'session' in m['metadata'])
    return {"visitors": 0, "chats": len(sessions), "leads": 0}

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
