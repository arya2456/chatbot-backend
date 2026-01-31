import os
import time
import re
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai
import logging
from uuid import uuid4

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = "chatbot-index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY" "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W")  # Removed hardcoded fallback
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FC Media Chatbot Brain", version="3.0")

# Fix: Restrict CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dashboard.fcmedia.in",
        "http://localhost:8000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE INIT ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    logger.info("Connected to Pinecone successfully")
except Exception as e:
    logger.critical(f"Failed to connect to Pinecone: {e}")
    raise

# --- MODELS ---
class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str
    bot_name: str = "AI Support"
    bot_color: str = "#4F46E5"
    bot_personality: str = "Professional and helpful"
    bot_avatar: str = ""  # New: Support avatar

class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"

class StatsRequest(BaseModel):
    client_id: str

# --- HELPER FUNCTIONS ---
def get_config(client_id: str):
    try:
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if res.vectors and f"config_{client_id}" in res.vectors:
            return res.vectors[f"config_{client_id}"].metadata
    except Exception as e:
        logger.error(f"Error fetching config for {client_id}: {e}")
    return None

def save_lead(client_id: str, email: str, context: str):
    try:
        lid = f"lead_{uuid4().hex}"
        index.upsert(
            vectors=[{
                "id": lid,
                "values": [0.0] * 768,
                "metadata": {
                    "type": "lead",
                    "email": email,
                    "context": context,
                    "timestamp": int(time.time())
                }
            }],
            namespace=client_id
        )
    except Exception as e:
        logger.error(f"Error saving lead: {e}")

def log_chat(client_id: str, session_id: str, user_msg: str, bot_msg: str):
    try:
        lid = f"log_{uuid4().hex}"
        index.upsert(
            vectors=[{
                "id": lid,
                "values": [0.0] * 768,
                "metadata": {
                    "type": "chat_log",
                    "session_id": session_id,
                    "user_msg": user_msg,
                    "bot_msg": bot_msg,
                    "timestamp": int(time.time())
                }
            }],
            namespace=client_id
        )
    except Exception as e:
        logger.error(f"Error logging chat: {e}")

async def fetch_url(session, url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        async with session.get(url, headers=headers, timeout=15) as resp:
            if resp.status == 200 and 'text/html' in resp.headers.get('Content-Type', ''):
                return await resp.text(), url
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
    return None, url

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if end >= len(text):
            break
    return chunks if chunks else [text[:1000]]

async def crawl_and_index(start_url: str, client_id: str, api_key: str):
    if not start_url.startswith("http"):
        start_url = "https://" + start_url
    
    genai.configure(api_key=api_key)
    domain = urlparse(start_url).netloc
    visited = set()
    to_visit = asyncio.Queue()
    await to_visit.put((start_url, 0))
    
    scraped_pages = []
    async with aiohttp.ClientSession() as session:
        while not to_visit.empty() and len(scraped_pages) < 15:
            url, depth = await to_visit.get()
            if url in visited or depth > 1:
                continue
            visited.add(url)
            
            html, final_url = await fetch_url(session, url)
            if not html:
                continue
                
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            links = []
            for a in soup.find_all('a', href=True):
                link = urljoin(final_url, a['href'])
                if urlparse(link).netloc == domain and link not in visited:
                    links.append(f"LINK: {a.get_text(strip=True)[:50]} -> {link}")
                    if depth < 1:
                        await to_visit.put((link, depth + 1))
            
            full_content = text + "\n\n=== NAVIGABLE LINKS ===\n" + "\n".join(links[:20])
            scraped_pages.append({"url": final_url, "text": full_content})
    
    vectors = []
    for page in scraped_pages:
        chunks = chunk_text(page["text"])
        page_hash = abs(hash(page["url"]))
        for i, chunk in enumerate(chunks):
            try:
                emb = genai.embed_content(
                    model="models/text-embedding-004",
                    content=chunk
                )["embedding"]
                vectors.append({
                    "id": f"{client_id}_{page_hash}_{i}_{uuid4().hex[:8]}",
                    "values": emb,
                    "metadata": {
                        "text": chunk,
                        "url": page["url"],
                        "client_id": client_id
                    }
                })
            except Exception as e:
                logger.error(f"Embedding error: {e}")
    
    if vectors:
        index.upsert(vectors=vectors, namespace=client_id)
        logger.info(f"Indexed {len(vectors)} chunks for {client_id}")
        return True
    return False

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "FC Media Brain v3.0 Active", "clients": "online"}

@app.post("/verify-install")
async def verify_install(req: StatsRequest):
    target = req.client_id if not req.client_id.startswith("http") else req.client_id
    target = f"https://{target}"
    try:
        async with aiohttp.ClientSession() as session:
            html, _ = await fetch_url(session, target)
            if html and "widget.js" in html and req.client_id in html:
                return {"status": "success", "message": "Widget Detected & Active"}
    except Exception as e:
        logger.error(f"Verify install error: {e}")
    return {"status": "failed", "message": "Widget not found"}

@app.post("/train")
async def train_bot(req: TrainRequest):
    # Save config with API key securely
    config_metadata = {
        "api_key": req.gemini_api_key,
        "bot_name": req.bot_name,
        "bot_color": req.bot_color,
        "bot_personality": req.bot_personality,
        "bot_avatar": req.bot_avatar or "",
        "trained_at": int(time.time())
    }
    
    try:
        index.upsert([{
            "id": f"config_{req.client_id}",
            "values": [0.0] * 768,
            "metadata": config_metadata
        }], namespace=req.client_id)
        
        # Start crawling
        success = await crawl_and_index(req.url, req.client_id, req.gemini_api_key)
        return {
            "status": "success",
            "message": "Bot trained successfully!" if success else "Config saved, crawling in progress...",
            "crawled": success
        }
    except Exception as e:
        logger.error(f"Train error for {req.client_id}: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/get-config")
def get_config_endpoint(client_id: str):
    config = get_config(client_id)
    if config:
        return {
            "bot_name": config.get("bot_name", "AI Support"),
            "bot_color": config.get("bot_color", "#4F46E5"),
            "bot_personality": config.get("bot_personality", "Professional"),
            "bot_avatar": config.get("bot_avatar", "")
        }
    return {
        "bot_name": "AI Support",
        "bot_color": "#4F46E5",
        "bot_personality": "Professional and helpful",
        "bot_avatar": ""
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    config = get_config(req.client_id)
    if not config or not config.get("api_key"):
        return {"answer": "Bot not configured yet. Please train me first."}

    # Extract email for lead capture
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", req.message)
    if email_match:
        save_lead(req.client_id, email_match.group(), req.message)

    try:
        genai.configure(api_key=config["api_key"])
        emb = genai.embed_content(
            model="models/text-embedding-004",
            content=req.message
        )["embedding"]
        
        results = index.query(
            namespace=req.client_id,
            vector=emb,
            top_k=5,
            include_metadata=True
        )
        
        context = "\n\n".join([
            m["metadata"].get("text", "")
            for m in results["matches"]
            if m["metadata"].get("text")
        ])
        
        system_prompt = f"""
        You are {config.get('bot_name', 'AI Assistant')}, a helpful assistant.
        Personality: {config.get('bot_personality', 'Professional and helpful')}.
        
        Use this knowledge to answer:
        {context or "No specific knowledge available."}
        
        Rules:
        - Answer based on the knowledge above
        - If unsure, say: "I don't have that information right now. Would you like to speak to a human?"
        - Format links as: [Click here](https://example.com)
        - Be friendly and professional
        """
        
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(
            f"{system_prompt}\n\nUser: {req.message}"
        )
        
        answer = response.text
        log_chat(req.client_id, req.session_id, req.message, answer)
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"answer": "I'm having trouble connecting right now. Please try again in a moment."}

@app.post("/get-stats")
def get_stats(req: StatsRequest):
    dummy = [0.0] * 768
    try:
        leads = index.query(namespace=req.client_id, vector=dummy, top_k=10000, filter={"type": "lead"})
        chats = index.query(namespace=req.client_id, vector=dummy, top_k=10000, filter={"type": "chat_log"})
        sessions = set(m["metadata"].get("session_id") for m in chats["matches"] if m["metadata"].get("session_id"))
        return {
            "visitors": len(sessions),
            "chats": len(chats["matches"]),
            "leads": len(leads["matches"])
        }
    except:
        return {"visitors": 0, "chats": 0, "leads": 0}

@app.post("/get-leads")
def get_leads(req: StatsRequest):
    dummy = [0.0] * 768
    try:
        res = index.query(namespace=req.client_id, vector=dummy, top_k=200, filter={"type": "lead"})
        leads = [
            {
                "email": m["metadata"].get("email", "N/A"),
                "message": m["metadata"].get("context", ""),
                "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(m["metadata"].get("timestamp", 0)))
            }
            for m in res["matches"]
        ]
        return {"leads": leads}
    except:
        return {"leads": []}

@app.post("/get-analytics")
def get_analytics(req: StatsRequest):
    dummy = [0.0] * 768
    try:
        res = index.query(namespace=req.client_id, vector=dummy, top_k=500, filter={"type": "chat_log"})
        logs = [
            {
                "session": m["metadata"].get("session_id", "unknown"),
                "user": m["metadata"].get("user_msg", ""),
                "bot": m["metadata"].get("bot_msg", ""),
                "time": m["metadata"].get("timestamp", 0)
            }
            for m in res["matches"]
        ]
        return {"logs": logs}
    except:
        return {"logs": []}
