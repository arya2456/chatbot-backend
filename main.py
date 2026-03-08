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
from datetime import datetime

# --- Import your new scraper module ---
import scraper 

# --- 1. CONFIG ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")
PHP_DASHBOARD_URL = "https://dashboard.fcmedia.in/api.php" 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

crawl_status_db = {}
session_memory = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Omni-Brain v29.0 - E-Com Edition", lifespan=lifespan)
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
    
    # --- UNIVERSAL E-COMMERCE INTEGRATION ---
    ecom_platform: str = ""  # e.g., "woocommerce", "shopify", "none"
    ecom_url: str = ""       # e.g., "https://clientstore.com"
    ecom_key: str = ""       # e.g., ck_12345 or shpat_12345
    ecom_secret: str = ""    # e.g., cs_12345

class ChatRequest(BaseModel):
    message: str; client_id: str; session_id: str = "Guest"; page_url: str = ""; api_key: str = ""

class StatusRequest(BaseModel):
    client_id: str; message: Optional[str] = None; session_id: Optional[str] = None
    api_key: Optional[str] = None; page_url: Optional[str] = None

# --- 3. HELPERS & SKILLS ---
def get_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

def safe_embed(text, api_key):
    if not api_key or len(api_key) < 10:
        logger.error("Gemini Embed Error: Missing or invalid API key.")
        return [0.0] * 768
        
    genai.configure(api_key=api_key)
    try: 
        emb = genai.embed_content(model="models/gemini-embedding-001", content=text)['embedding']
        if len(emb) > 768:
            return emb[:768]
        return emb
    except Exception as e: 
        logger.error(f"Gemini Embed Error: {str(e)}")
        return [0.0] * 768

# E-COMMERCE API SKILL
def check_ecommerce_order(platform: str, url: str, key: str, secret: str, order_id: str):
    if not url or not key:
        return "Error: Store credentials are not fully configured in the dashboard."
        
    platform = platform.lower().strip()
    
    if platform == "woocommerce":
        # WooCommerce REST API: GET /wp-json/wc/v3/orders/<id>
        api_url = urljoin(url, f"/wp-json/wc/v3/orders/{order_id}")
        try:
            # WooCommerce uses basic auth for the consumer key/secret
            resp = requests.get(api_url, auth=(key, secret), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "Unknown").title()
                total = data.get("total", "0.00")
                currency = data.get("currency", "")
                
                # Format a friendly summary for Gemini to read
                return f"SUCCESS: Order {order_id} was found. Current Status: {status}. Order Total: {total} {currency}."
            elif resp.status_code == 404:
                return f"NOT FOUND: Order {order_id} does not exist in the system."
            else:
                return f"ERROR: Failed to retrieve order. API Status Code {resp.status_code}."
        except Exception as e:
            return f"API ERROR: Could not connect to the store database: {str(e)}"
            
    elif platform == "shopify":
        # Placeholder for future Shopify logic
        return "Shopify tracking is coming soon."
        
    return "Error: E-commerce platform not recognized."

# --- The Silent SDR (Lead Extractor) ---
async def extract_and_save_lead(user_msg: str, client_id: str, api_key: str):
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
        
        if lead_data.get("email") or lead_data.get("phone"):
            payload = {
                "client_id": client_id,
                "email": lead_data.get("email") or "Unknown",
                "phone": lead_data.get("phone") or "",
                "name": lead_data.get("name") or "Website Visitor",
                "company": lead_data.get("company") or "",
                "message": user_msg 
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            requests.post(f"{PHP_DASHBOARD_URL}?action=save_lead", json=payload, headers=headers, timeout=5)
            logger.info(f"💰 SILENT LEAD CAPTURED: {lead_data.get('email') or lead_data.get('phone')}")
            
    except Exception as e:
        logger.error(f"Lead extraction failed silently: {e}")

# --- The Heavy Engine Room Task for Scrapping ---
async def perform_deep_sync(client_id: str, url: str, api_key: str):
    crawl_status_db[client_id] = {"status": "crawling", "progress": 10, "pages": 0}
    try:
        pages_data = scraper.crawl_website(url, max_pages=200)
        crawl_status_db[client_id] = {"status": "processing", "progress": 50, "pages": len(pages_data)}
        
        all_chunks_info = []
        for page in pages_data:
            words = page['text'].split()
            chunks = [' '.join(words[i:i+200]) for i in range(0, len(words), 200)]
            
            for chunk in chunks:
                if len(chunk) > 20: 
                    context_rich_chunk = f"PAGE SOURCE: {page.get('title', 'Unknown Page')}\n{chunk}"
                    chunk_hash = hashlib.md5(context_rich_chunk.encode('utf-8')).hexdigest()
                    all_chunks_info.append({
                        "id": f"doc_{chunk_hash}",
                        "text": context_rich_chunk,
                        "url": page["url"]
                    })

        existing_ids = set()
        chunk_ids = [c["id"] for c in all_chunks_info]
        
        for i in range(0, len(chunk_ids), 500):
            try:
                res = index.fetch(ids=chunk_ids[i:i+500], namespace=client_id)
                existing_ids.update(res.vectors.keys())
            except Exception as e:
                logger.warning(f"Pinecone check failed, embedding all: {e}")

        vectors_to_upsert = []
        embedded_count = 0

        for chunk_data in all_chunks_info:
            if chunk_data["id"] not in existing_ids:
                emb = safe_embed(chunk_data["text"], api_key)
                if any(v != 0.0 for v in emb):
                    vectors_to_upsert.append({
                        "id": chunk_data["id"],
                        "values": emb,
                        "metadata": {"type": "knowledge", "text": chunk_data["text"], "url": chunk_data["url"]}
                    })
                    embedded_count += 1

        if vectors_to_upsert:
            batch_size = 50
            for i in range(0, len(vectors_to_upsert), batch_size):
                index.upsert(vectors=vectors_to_upsert[i:i+batch_size], namespace=client_id)
                
        crawl_status_db[client_id] = {"status": "complete", "progress": 100, "pages": len(pages_data)}
        logger.info(f"✅ Deep Sync Complete! Embedded: {embedded_count} | Skipped: {len(all_chunks_info) - embedded_count}")
        
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

        # 2. TRIGGER SILENT SALESMAN
        background_tasks.add_task(extract_and_save_lead, req.message, req.client_id, active_key)

        # 3. Fetch Knowledge
        emb = safe_embed(req.message, active_key)
        search = index.query(namespace=req.client_id, vector=emb, top_k=8, include_metadata=True, filter={"type": "knowledge"})
        
        context = ""
        for match in search['matches']:
            text = match['metadata'].get('text', '')
            url = match['metadata'].get('url', 'No link available')
            context += f"Content: {text}\nSource Link: {url}\n\n"
            
        # 4. Retrieve Short-Term Memory
        if req.session_id not in session_memory:
            session_memory[req.session_id] = []
        history_text = "\n".join(session_memory[req.session_id][-6:])
        
        # 5. UNIVERSAL SALES PROMPT WITH ADVANCED CAPABILITIES
        biz_contact = f"Contact: {conf.get('biz_phone', '')}, {conf.get('biz_email', '')}."
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_current_url = req.page_url if req.page_url else "Unknown"
        
        # --- FIX: GRAB THE CUSTOM RULES FROM THE DASHBOARD ---
        custom_rules = conf.get('bot_personality', 'Professional')
        
        sys_msg = f"""
        Role: You are {conf.get('bot_name')}, a highly intelligent, conversational AI representative for {conf.get('biz_name')}.
        Persona: {conf.get('bot_personality', 'Professional and helpful')}.
        Business Details: {biz_contact}
        System Time: {current_time}
        User's Current Webpage: {user_current_url}

        === CUSTOM CLIENT RULES (HIGHEST PRIORITY) ===
        You MUST strictly obey these specific instructions. If these rules contradict the general rules below, THESE CUSTOM RULES WIN:
        {custom_rules}

        === CORE DIRECTIVES ===
        1. CONTEXT IS KING: Read the 'Knowledge Base Context' carefully. Always answer the user's question directly using ONLY that information. If the answer is there, explain it naturally and conversationally.
        2. NO ROBOTIC REPETITION: Do not repeat the same exact phrases over and over. If the user changes the subject or ignores a question, adapt naturally.
        3. THE GRACEFUL PIVOT (LEAD CAPTURE): If you need to ask for contact info, make it smooth. 
           - NEVER blindly append a robotic "I don't have the exact link handy..." script if you already answered their question. 
           - If your Custom Rules tell you to ask for a WhatsApp number, ONLY ask for a WhatsApp number. Do not ask for an email.
        4. HANDLING MISSING INFO: Only if the Knowledge Base lacks the answer ENTIRELY, do not guess. Instead say: "That's a great question. I want to make sure I give you the 100% correct answer—can I have our team reach out to you with the exact details?"
        5. HOW TO FIND LINKS: If asked for a link, look INSIDE the "Content:" text for a URL. ONLY provide links explicitly written there. DO NOT guess or invent URLs.
        6. CLICKABLE HTML LINKS: Format valid URLs exactly like this: <a href="THE_EXACT_URL_HERE" target="_blank" style="color: #0ea5e9; text-decoration: underline; font-weight: bold;">Click here</a>
        7. MEMORY AWARENESS: Read the 'Recent Chat History'. If the user has already provided their contact info (email/phone/WhatsApp), NEVER ask for it again.
        
        8. ORDER TRACKING (NEW SKILL): If the user asks to track an order AND provides the order number, STOP TALKING and output EXACTLY: [CHECK_ORDER: their_order_number]. If they don't provide a number, politely ask for it.

        Knowledge Base Context (Your ONLY source of truth):
        {context}

        Recent Chat History (Do not repeat yourself):
        {history_text}

        Current User Message: {req.message}
        """
        
        ans = get_model(active_key).generate_content(sys_msg).text
        
        # --- THE E-COMMERCE INTERCEPTOR ---
        if "[CHECK_ORDER:" in ans:
            # Extract the ID from the string "[CHECK_ORDER: 12345]"
            match = re.search(r'\[CHECK_ORDER:\s*#?([a-zA-Z0-9_-]+)\]', ans)
            if match:
                order_id = match.group(1)
                
                # Retrieve client's E-Commerce keys from Pinecone memory
                platform = conf.get("ecom_platform", "")
                store_url = conf.get("ecom_url", "")
                store_key = conf.get("ecom_key", "")
                store_secret = conf.get("ecom_secret", "")
                
                logger.info(f"Intercepted Order Request for ID: {order_id}. Checking Platform: {platform}")
                
                # Dip into WooCommerce API
                raw_order_data = check_ecommerce_order(platform, store_url, store_key, store_secret, order_id)
                
                # Feed the raw API data back to Gemini to translate into a friendly message
                follow_up_prompt = f"""
                You are the AI assistant. The user just asked to track order #{order_id}.
                We securely checked the store database and received this raw data: {raw_order_data}
                
                Translate this raw data into a polite, helpful message for the customer. DO NOT output the [CHECK_ORDER] tag again.
                """
                ans = get_model(active_key).generate_content(follow_up_prompt).text
            else:
                ans = "I'm having trouble reading that order number. Could you please double-check it and type it again?"
        # --- END E-COMMERCE INTERCEPTOR ---

        # 6. Save to Short-Term Memory
        session_memory[req.session_id].append(f"User: {req.message}")
        session_memory[req.session_id].append(f"Bot: {ans}")
        if len(session_memory[req.session_id]) > 10:
            session_memory[req.session_id] = session_memory[req.session_id][-10:]
        
        # 7. Save Logs to Pinecone & MySQL
        log_meta = {"type": "chat_log", "session": req.session_id, "user_msg": req.message, "bot_msg": ans, "timestamp": int(time.time())}
        index.upsert(vectors=[{"id": f"log_{uuid.uuid4()}", "values": [0.1]*768, "metadata": log_meta}], namespace=req.client_id)
        
        try:
            payload = {
                "client_id": req.client_id,
                "session_id": req.session_id,
                "user_msg": req.message,
                "bot_msg": ans
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            requests.post(f"{PHP_DASHBOARD_URL}?action=save_chat", json=payload, headers=headers, timeout=3)
        except Exception as db_err:
            logger.error(f"FATAL: Could not sync to MySQL: {db_err}")

        return {"answer": ans}
    except Exception as e: return {"answer": f"System error: {str(e)}"}

# --- THE NEW LEAD PUSHER ---
@app.post("/capture-lead")
async def capture_lead(req: dict):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        db_response = requests.post(f"{PHP_DASHBOARD_URL}?action=save_lead", json=req, headers=headers, timeout=3)
        logger.info(f"MySQL Lead Sync Status: {db_response.status_code}")
        
        log_meta = req.copy()
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

# --- Smarter Train Endpoint to support Scrapping ---
@app.post("/train")
async def train_saas_engine(req: TrainRequest, background_tasks: BackgroundTasks):
    try:
        res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
        existing_conf = res.vectors[f"config_{req.client_id}"].metadata if res.vectors else {}
    except:
        existing_conf = {}

    incoming_data = req.model_dump(exclude_unset=True) if hasattr(req, 'model_dump') else req.dict(exclude_unset=True)
    
    meta = existing_conf.copy()
    meta.update(incoming_data)
    meta["type"] = "config"
    
    index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
    
    active_key = meta.get("gemini_api_key", req.gemini_api_key)
    if incoming_data.get("url"):
        background_tasks.add_task(perform_deep_sync, req.client_id, meta.get("url", ""), active_key)
        
    return {"status": "success"}

# --- The Endpoint that talks to the Dashboard Progress Bar ---
@app.post("/get-crawl-status")
async def get_crawl_status(req: StatusRequest):
    return crawl_status_db.get(req.client_id, {"status": "idle", "progress": 0, "pages": 0})

# --- MISSING DASHBOARD FEATURE: Document Upload ---
@app.post("/upload-file")
async def upload_document(client_id: str, file: UploadFile = File(...)):
    try:
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        conf = res.vectors[f"config_{client_id}"].metadata if res.vectors else {}
        api_key = conf.get("gemini_api_key", "")
        if not api_key: return {"status": "error", "message": "API Key missing."}

        content = ""
        if file.filename.lower().endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(io.BytesIO(await file.read()))
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        else:
            content = (await file.read()).decode("utf-8")

        words = content.split()
        chunks = [' '.join(words[i:i+200]) for i in range(0, len(words), 200)]
        
        vectors = []
        for chunk in chunks:
            if len(chunk) > 20:
                emb = safe_embed(chunk, api_key)
                if any(v != 0.0 for v in emb):
                    vectors.append({
                        "id": f"doc_{uuid.uuid4()}",
                        "values": emb,
                        "metadata": {"type": "knowledge", "text": chunk, "url": f"Document: {file.filename}"}
                    })

        if vectors:
            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                index.upsert(vectors=vectors[i:i+batch_size], namespace=client_id)

        # --- FIX: REPORT UPLOAD TO DASHBOARD MYSQL ---
        try:
            payload = { "client_id": client_id, "filename": file.filename }
            headers = {"Content-Type": "application/json"}
            requests.post(f"{PHP_DASHBOARD_URL}?action=save_document", json=payload, headers=headers, timeout=3)
        except Exception as db_e:
            logger.error(f"Could not report document to PHP: {db_e}")

        return {"status": "success", "filename": file.filename}
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/")
def health(): return {"status": "Omni-Brain v29.0 - E-Com Support Active"}
