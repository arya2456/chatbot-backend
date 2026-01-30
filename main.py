import os
import asyncio
import aiohttp
import time
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai
import xml.etree.ElementTree as ET

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = "chatbot-index"
# This is the ONLY key you pay for (Database storage), which is tiny/free.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W")

app = FastAPI(title="FC Brain Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Database Init Error: {e}")

# --- MODELS ---
class TrainRequest(BaseModel):
    url: str
    client_id: str
    gemini_api_key: str  # <--- THIS IS THE CLIENT'S KEY FROM DASHBOARD
    bot_name: str = "AI Support"
    bot_color: str = "#4F46E5"
    bot_avatar: str = ""

class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest-Unknown" 

class AutoSyncRequest(BaseModel):
    url: str
    client_id: str

# --- HELPERS ---
def save_client_config(client_id, api_key, bot_name, bot_color, bot_avatar):
    """Saves the Client's API Key into the Database securely"""
    try:
        index.upsert(
            vectors=[{
                "id": f"config_{client_id}",
                "values": [1.0] * 768, 
                "metadata": {
                    "api_key": api_key,  # Storing Client's specific key
                    "type": "config",
                    "bot_name": bot_name,
                    "bot_color": bot_color,
                    "bot_avatar": bot_avatar
                }
            }],
            namespace=client_id
        )
    except Exception as e:
        print(f"Config Error: {e}")

def get_client_config(client_id):
    """Retrieves the specific Client's API Key"""
    try:
        response = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if f"config_{client_id}" in response.vectors:
            return response.vectors[f"config_{client_id}"].metadata
        return None
    except:
        return None

def check_and_save_lead(message, client_id):
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_regex, message)
    if match:
        email = match.group()
        timestamp = int(time.time())
        lead_id = f"lead_{timestamp}_{abs(hash(email))}"
        try:
            index.upsert(
                vectors=[{
                    "id": lead_id,
                    "values": [1.0] * 768,
                    "metadata": {"type": "lead", "email": email, "context": message, "timestamp": timestamp}
                }],
                namespace=client_id
            )
            return True
        except: return False
    return False

def log_chat(client_id, session_id, user_msg, bot_msg):
    try:
        log_id = f"chat_{session_id}_{int(time.time())}"
        index.upsert(
            vectors=[{
                "id": log_id,
                "values": [0.1] * 768,
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
        print(f"Log Error: {e}")

# --- DYNAMIC EMBEDDING HELPER ---
def get_embedding(text: str, client_api_key: str, task_type: str = "retrieval_document"):
    """Uses the CLIENT'S specific API Key to generate embeddings"""
    # 1. Switch to Client's Key
    genai.configure(api_key=client_api_key)
    
    # 2. Use Safe Model
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type=task_type
    )
    return result['embedding']

def get_best_model():
    return "models/gemini-1.5-flash"

# --- CRAWLER LOGIC ---
async def fetch_sitemap(session, base_url):
    potential_sitemaps = [urljoin(base_url, "sitemap.xml"), urljoin(base_url, "wp-sitemap.xml")]
    found_urls = set()
    for sitemap_url in potential_sitemaps:
        try:
            async with session.get(sitemap_url, timeout=10) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    try:
                        root = ET.fromstring(content)
                        for elem in root.iter():
                            if 'loc' in elem.tag and elem.text: found_urls.add(elem.text.strip())
                        if found_urls: return list(found_urls)
                    except: pass
        except: pass
    return []

async def fetch_url(session, url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FC-Bot/1.0)"}
    try:
        async with session.get(url, headers=headers, timeout=12) as resp:
            if resp.status == 200:
                return await resp.text(), url
    except: pass
    return None, url

def smart_chunk_text(text, max_chars=3000):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current += para + "\n"
        else:
            if current: chunks.append(current.strip())
            current = para + "\n"
    if current: chunks.append(current.strip())
    return chunks

async def crawl_and_index(url, client_id, api_key, bot_name, bot_color, bot_avatar):
    # 1. Save the CLIENT'S Key to the DB
    save_client_config(client_id, api_key, bot_name, bot_color, bot_avatar)
    
    if not url.startswith('http'): url = 'https://' + url
    
    async with aiohttp.ClientSession() as session:
        sitemap_urls = await fetch_sitemap(session, url)
        to_visit = set(sitemap_urls[:60]) if sitemap_urls else {url}
        visited = set()
        queue = list(to_visit)
        scraped_data = []
        
        while queue and len(visited) < 60:
            batch = queue[:8]
            queue = queue[8:]
            results = await asyncio.gather(*[fetch_url(session, u) for u in batch])

            for html, current_url in results:
                if not html: continue
                visited.add(current_url)
                soup = BeautifulSoup(html, 'html.parser')
                
                if not sitemap_urls:
                    base_domain = urlparse(url).netloc.replace("www.", "")
                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(current_url, link['href']).split('#')[0].split('?')[0]
                        if (base_domain in full_url and full_url not in visited and full_url not in queue
                                and not any(x in full_url for x in ['.jpg', '.png', '.pdf', 'login', 'admin'])):
                            queue.append(full_url)

                for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript", "header"]): tag.extract()
                text = soup.get_text(separator='\n', strip=True)
                if len(text) > 250: scraped_data.append({"url": current_url, "text": text})

    if not scraped_data: return False

    try:
        # Use CLIENT'S Key for Embedding
        vectors = []
        for page in scraped_data:
            chunks = smart_chunk_text(page['text'])
            for i, chunk in enumerate(chunks):
                embedding = get_embedding(text=chunk, client_api_key=api_key, task_type="retrieval_document")
                vector_id = f"{client_id}_{abs(hash(page['url']))}_{i}"
                vectors.append({"id": vector_id, "values": embedding, "metadata": {"text": chunk, "url": page['url']}})

        if vectors:
            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                index.upsert(vectors=vectors[i:i + batch_size], namespace=client_id)
            index.upsert(vectors=[{"id": "config_SYNC", "values": [0.0] * 768, "metadata": {"last_sync_timestamp": int(time.time())}}], namespace=client_id)
            return True
    except Exception as e:
        print(f"Indexing Error: {e}")
        return False

# --- API ENDPOINTS ---

@app.get("/")
def home(): return {"status": "FC Brain Active (Multi-Tenant BYOK)"}

@app.post("/train")
async def train_bot(request: TrainRequest):
    # This receives the key from the Dashboard and saves it
    success = await crawl_and_index(
        request.url, request.client_id, request.gemini_api_key, 
        request.bot_name, request.bot_color, request.bot_avatar
    )
    return {"status": "success" if success else "failed"}

@app.get("/get-config")
async def get_config(client_id: str):
    config = get_client_config(client_id)
    if config:
        return {
            "bot_name": config.get("bot_name", "AI Support"),
            "bot_color": config.get("bot_color", "#4F46E5"),
            "bot_avatar": config.get("bot_avatar", "") 
        }
    return {"bot_name": "Support", "bot_color": "#4F46E5", "bot_avatar": ""}

@app.post("/chat")
async def chat_bot(request: ChatRequest):
    # 1. Fetch THIS client's config from DB
    config = get_client_config(request.client_id)
    if not config: return {"answer": "Error: Bot not configured."}
    
    # 2. Extract THIS client's API Key
    client_api_key = config.get("api_key")
    if not client_api_key: return {"answer": "Error: Client API Key missing."}

    is_lead = check_and_save_lead(request.message, request.client_id)

    try:
        # 3. Configure Google to use Client's Key
        genai.configure(api_key=client_api_key)
        
        # 4. Generate Embedding (using Client Key)
        embedding = get_embedding(text=request.message, client_api_key=client_api_key, task_type="retrieval_query")
        
        search_results = index.query(namespace=request.client_id, vector=embedding, top_k=5, include_metadata=True)
        context = "\n\n".join([f"SOURCE: {m['metadata'].get('url','')}\nTEXT: {m['metadata']['text']}" for m in search_results['matches']])
        
        system = f"You are a helpful assistant for {request.client_id}. CONTEXT: {context}. INSTRUCTIONS: Answer strictly based on context. Be concise."
        if is_lead: system += "\n(User provided email. Confirm receipt.)"

        model = genai.GenerativeModel(get_best_model())
        response = model.generate_content(f"{system}\n\nUSER: {request.message}")
        
        log_chat(request.client_id, request.session_id, request.message, response.text)
        return {"answer": response.text}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

# --- RESTORED ENDPOINT: LEADS ---
@app.post("/get-leads")
async def get_leads(request: AutoSyncRequest):
    try:
        dummy = [0.1] * 768
        results = index.query(namespace=request.client_id, vector=dummy, top_k=100, include_metadata=True, filter={"type": "lead"})
        leads = []
        for m in results['matches']:
            leads.append({
                "email": m['metadata'].get('email'),
                "message": m['metadata'].get('context'),
                "date": m['metadata'].get('timestamp')
            })
        return {"leads": leads}
    except: return {"leads": []}

# --- RESTORED ENDPOINT: ANALYTICS ---
@app.post("/get-analytics")
async def get_analytics(request: AutoSyncRequest):
    try:
        config = get_client_config(request.client_id)
        if not config: return {"logs": [], "summary": "No data"}
        
        dummy = [0.1] * 768
        results = index.query(namespace=request.client_id, vector=dummy, top_k=100, include_metadata=True, filter={"type": "chat_log"})
        
        logs = []
        user_questions = []
        for m in results['matches']:
            logs.append({
                "session": m['metadata'].get('session_id'),
                "user": m['metadata'].get('user_msg'),
                "bot": m['metadata'].get('bot_msg'),
                "time": m['metadata'].get('timestamp')
            })
            user_questions.append(m['metadata'].get('user_msg'))

        ai_summary = "Not enough data yet."
        if len(user_questions) > 5:
            try:
                # Use CLIENT'S KEY for Analytics too
                genai.configure(api_key=config.get("api_key"))
                model = genai.GenerativeModel(get_best_model())
                res = model.generate_content(f"Analyze these user questions and list Top 3 common topics:\n{', '.join(user_questions[:30])}")
                ai_summary = res.text
            except: ai_summary = "Analysis unavailable."

        return {"logs": logs, "summary": ai_summary}
    except Exception as e:
        return {"logs": [], "summary": f"Error: {str(e)}"}
