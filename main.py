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

# --- ADDED: Import your new scraper module ---
import scraper 

# --- 1. CONFIG ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")
# This points the AI to your new MySQL Brain!
PHP_DASHBOARD_URL = "https://dashboard.fcmedia.in/api.php" 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ADDED: A temporary memory bank to track crawl progress for the dashboard ---
crawl_status_db = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Omni-Brain v28.0", lifespan=lifespan)
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
    if not api_key or len(api_key) < 10:
        logger.error("Gemini Embed Error: Missing or invalid API key.")
        return [0.0] * 768
        
    genai.configure(api_key=api_key)
    try: 
        # 1. Generate the embedding (It will be 3072 dimensions)
        emb = genai.embed_content(model="models/gemini-embedding-001", content=text)['embedding']
        
        # 2. CRITICAL FIX: Slice it down to 768 to fit your database
        if len(emb) > 768:
            return emb[:768]
            
        return emb
    except Exception as e: 
        logger.error(f"Gemini Embed Error: {str(e)}")
        return [0.0] * 768

# --- ADDED: The Silent SDR (Lead Extractor) ---
async def extract_and_save_lead(user_msg: str, client_id: str, api_key: str):
    """Silently scans messages for contact info and pushes to MySQL CRM."""
    # Fast check: Only run AI if there's an '@' symbol or a cluster of numbers
    if "@" not in user_msg and len(re.findall(r'\d', user_msg)) < 7:
        return

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        Analyze this text: "{user_msg}". 
        Extract any names, emails, phone numbers, or company names mentioned.
        Return ONLY a raw JSON object with keys: "name", "email", "phone", "company". 
        If a field is missing, use null. Do not write markdown or code blocks, just the JSON string.
        """
        resp = model.generate_content(prompt)
        cleaned_json = resp.text.replace("```json", "").replace("```", "").strip()
        lead_data = json.loads(cleaned_json)
        
        # If actionable data is found, push to Dashboard
        if lead_data.get("email") or lead_data.get("phone"):
            payload = {
                "client_id": client_id,
                "email": lead_data.get("email") or "Unknown",
                "phone": lead_data.get("phone") or "",
                "name": lead_data.get("name") or "Website Visitor",
                "company": lead_data.get("company") or "",
                "message": user_msg 
            }
            # Add Hostinger bypass headers just in case
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            requests.post(f"{PHP_DASHBOARD_URL}?action=save_lead", json=payload, headers=headers, timeout=5)
            logger.info(f"💰 SILENT LEAD CAPTURED: {lead_data.get('email') or lead_data.get('phone')}")
            
    except Exception as e:
        logger.error(f"Lead extraction failed silently: {e}")

# --- ADDED: The Heavy Engine Room Task for Scrapping ---
async def perform_deep_sync(client_id: str, url: str, api_key: str):
    crawl_status_db[client_id] = {"status": "crawling", "progress": 10, "pages": 0}
    try:
        # 1. Start the Scrape!
        pages_data = scraper.crawl_website(url, max_pages=15)
        crawl_status_db[client_id] = {"status": "crawling", "progress": 50, "pages": len(pages_data)}
        
        # 2. Chop text into AI-friendly chunks (Vectors)
        vectors = []
        for page in pages_data:
            words = page['text'].split()
            # Split page into chunks of ~200 words
            chunks = [' '.join(words[i:i+200]) for i in range(0, len(words), 200)]
            
            for chunk in chunks:
                if len(chunk) > 20: # Ignore tiny useless chunks
                    emb = safe_embed(chunk, api_key)
                    # FIX: Only append if the vector has valid data (not all zeros)
                    if any(v != 0.0 for v in emb):
                        vectors.append({
                            "id": f"doc_{uuid.uuid4()}",
                            "values": emb,
                            "metadata": {"type": "knowledge", "text": chunk, "url": page["url"]}
                        })
                    else:
                        logger.warning("Skipped a chunk because Gemini failed to embed it.")
        
        # 3. Push to Pinecone in safe batches
        if vectors:
            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                index.upsert(vectors=vectors[i:i+batch_size], namespace=client_id)
                
        crawl_status_db[client_id] = {"status": "complete", "progress": 100, "pages": len(pages_data)}
        logger.info(f"Deep sync complete for {client_id}")
    except Exception as e:
        logger.error(f"Deep sync failed: {str(e)}")
        crawl_status_db[client_id] = {"status": "error", "progress": 0, "pages": 0}

# --- 4. ENGINE ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    try:
        # 1. Get Config & Context
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        conf = res.vectors[f"config_{req.client_id}"].metadata if res.vectors else {}
        active_key = req.api_key if len(req.api_key) > 10 else conf.get("gemini_api_key", "")
        if not active_key: return {"answer": "API Key missing."}

        # 2. TRIGGER SILENT SALESMAN (Runs in background so it doesn't slow down the chat)
        background_tasks.add_task(extract_and_save_lead, req.message, req.client_id, active_key)

        # 3. Fetch Knowledge
        emb = safe_embed(req.message, active_key)
        search = index.query(namespace=req.client_id, vector=emb, top_k=3, include_metadata=True, filter={"type": "knowledge"})
        context = "\n".join([m['metadata']['text'] for m in search['matches']])
        
        # 4. UNIVERSAL SALES PROMPT
        biz_contact = f"Contact: {conf.get('biz_phone', '')}, {conf.get('biz_email', '')}."
        
        sys_msg = f"""
        Role: You are {conf.get('bot_name')}, a professional AI assistant for {conf.get('biz_name')}.
        Personality: {conf.get('bot_personality')}.
        Business Details: {biz_contact}

        CRITICAL SALES RULES (UNIVERSAL SDR):
        1. CONVERSATIONAL DRIP: If the user asks about pricing, buying, or booking, do not give away everything at once. Naturally ask for their Name or Email first to "send them the details" or "check availability".
        2. THE KNOWLEDGE GAP: If the user asks a specific question NOT covered in the Context below, do NOT guess. Tell them it's a great question for a specialist and ask for their email or phone number so an expert can reach out.
        3. ASSUME THE CLOSE: Never end a message with a dead-end statement. Always end with a polite, relevant question that keeps the conversation moving toward capturing their contact info or moving to the next step.

        Knowledge Base Context:
        {context}

        User Message: {req.message}
        """
        
        ans = get_model(active_key).generate_content(sys_msg).text
        
        # 5. Save Logs to Pinecone & MySQL
        log_meta = {"type": "chat_log", "session": req.session_id, "user_msg": req.message, "bot_msg": ans, "timestamp": int(time.time())}
        index.upsert(vectors=[{"id": f"log_{uuid.uuid4()}", "values": [0.1]*768, "metadata": log_meta}], namespace=req.client_id)
        
        try:
            payload = {
                "client_id": req.client_id,
                "session_id": req.session_id,
                "user_msg": req.message,
                "bot_msg": ans
            }
            # --- Added Headers to Bypass Hostinger 403 Forbidden ---
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            # Using synchronous requests for guaranteed delivery with headers
            db_response = requests.post(f"{PHP_DASHBOARD_URL}?action=save_chat", json=payload, headers=headers, timeout=3)
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
        # --- Added Headers to Bypass Hostinger 403 Forbidden ---
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        # Force a synchronous push to bypass the firewall rules with headers
        db_response = requests.post(f"{PHP_DASHBOARD_URL}?action=save_lead", json=req, headers=headers, timeout=3)
        logger.info(f"MySQL Lead Sync Status: {db_response.status_code}")
        
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

# --- UPDATED: Smarter Train Endpoint to support Scrapping ---
@app.post("/train")
async def train_saas_engine(req: TrainRequest, background_tasks: BackgroundTasks):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        existing_conf = res.vectors[f"config_{req.client_id}"].metadata if res.vectors else {}
    except:
        existing_conf = {}

    # Extract only the data sent by the dashboard, avoiding defaults from overwriting your config
    incoming_data = req.model_dump(exclude_unset=True) if hasattr(req, 'model_dump') else req.dict(exclude_unset=True)
    
    meta = existing_conf.copy()
    meta.update(incoming_data)
    meta["type"] = "config"
    
    # Ensuring the config vector matches the index dimension (768)
    index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
    
    # If the user pushed the "Deep Sync Site" button, trigger the scraper!
    active_key = meta.get("gemini_api_key", req.gemini_api_key)
    if incoming_data.get("url"):
        background_tasks.add_task(perform_deep_sync, req.client_id, meta.get("url", ""), active_key)
        
    return {"status": "success"}

# --- ADDED: The Endpoint that talks to the Dashboard Progress Bar ---
@app.post("/get-crawl-status")
async def get_crawl_status(req: StatusRequest):
    return crawl_status_db.get(req.client_id, {"status": "idle", "progress": 0, "pages": 0})

@app.get("/")
def health(): return {"status": "Omni-Brain v28.0 - Universal SDR Active"}
