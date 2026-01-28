import os
import asyncio
import aiohttp
import time
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = "chatbot-index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W")

app = FastAPI(title="FC Brain Chatbot")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE SETUP ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"❌ Pinecone Error: {e}")
    raise e  # Stop server if Pinecone fails

# --- REQUEST MODELS ---
class TrainRequest(BaseModel):
    url: str
    client_id: str
    gemini_api_key: str
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

# ---------------------------------
# ---------- HELPERS ---------------
# ---------------------------------

def save_client_config(client_id, api_key, bot_name, bot_color, bot_avatar):
    """Saves client's bot config to Pinecone"""
    try:
        index.upsert(
            vectors=[{
                "id": f"config_{client_id}",
                "values": [1.0] * 768,
                "metadata": {
                    "api_key": api_key,
                    "type": "config",
                    "bot_name": bot_name,
                    "bot_color": bot_color,
                    "bot_avatar": bot_avatar
                }
            }],
            namespace=client_id
        )
        print(f"✅ Config saved for {client_id}")
    except Exception as e:
        print(f"❌ Config Save Error: {e}")
        raise

def get_client_config(client_id):
    """Fetches client's bot config. **CRITICAL FOR DASHBOARD!**"""
    try:
        response = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if f"config_{client_id}" in response.vectors:
            return response.vectors[f"config_{client_id}"].metadata
        return None
    except Exception as e:
        print(f"❌ Get Config Error: {e}")
        return None

def check_and_save_lead(message, client_id):
    """Detects email & saves as lead"""
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_regex, message)
    if not match:
        return False
    
    email = match.group()
    timestamp = int(time.time())
    lead_id = f"lead_{timestamp}_{abs(hash(email))}"
    
    try:
        index.upsert(
            vectors=[{
                "id": lead_id,
                "values": [1.0] * 768,
                "metadata": {
                    "type": "lead",
                    "email": email,
                    "context": message,
                    "timestamp": timestamp
                }
            }],
            namespace=client_id
        )
        print(f"💌 Lead saved: {email}")
        return True
    except Exception as e:
        print(f"❌ Lead Save Error: {e}")
        return False

def log_chat(client_id, session_id, user_msg, bot_msg):
    """Logs chat history"""
    try:
        log_id = f"chat_{session_id}_{int(time.time())}"
        index.upsert(vectors=[{
            "id": log_id,
            "values": [0.1] * 768,
            "metadata": {
                "type": "chat_log",
                "session_id": session_id,
                "user_msg": user_msg,
                "bot_msg": bot_msg,
                "timestamp": int(time.time())
            }
        }], namespace=client_id)
    except Exception as e:
        print(f"⚠️ Log Error: {e}")

# ---------------------------------
# --- EMBEDDING HELPER (FIXED!) ----
# ---------------------------------
def get_embedding(text: str, task_type: str = "retrieval_document", title: str = None):
    """
    ✅ USES text-embedding-004 (NEW) 
    🔄 Falls back to embedding-001 ONLY if 004 fails (for compatibility)
    """
    models_to_try = [
        "models/text-embedding-004",   # ✅ PREFERRED (NEWEST)
        "models/embedding-001"         # 🔄 FALLBACK (OLD - only if 004 fails)
    ]
    
    for model_name in models_to_try:
        try:
            result = genai.embed_content(
                model=model_name,
                content=text,
                task_type=task_type,
                title=title
            )
            print(f"📌 Using embedding model: {model_name}")  # DEBUG INFO
            return result['embedding']
        except Exception as e:
            print(f"⚠️ Model '{model_name}' failed: {e}. Trying next...")
    
    # If ALL models fail
    raise RuntimeError("❌ ALL embedding models failed! Check API key & model access.")

# ---------------------------------
# -------- CRAWLER -----------------
# ---------------------------------

async def fetch_sitemap(session, base_url):
    """Fetches sitemap URLs"""
    sitemap_urls = [
        urljoin(base_url, "sitemap.xml"),
        urljoin(base_url, "sitemap_index.xml"),
        urljoin(base_url, "wp-sitemap.xml")
    ]
    found_urls = set()
    
    for sitemap_url in sitemap_urls:
        try:
            async with session.get(sitemap_url, timeout=10) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    root = ET.fromstring(content)
                    for elem in root.iter():
                        if 'loc' in elem.tag and elem.text:
                            found_urls.add(elem.text.strip())
                    if found_urls: 
                        return list(found_urls)
        except:
            continue
    return []

async def fetch_url(session, url):
    """Fetches a single URL"""
    headers = {"User-Agent": "FC-Bot/1.0 (+https://yourdomain.com/bot)"}
    try:
        async with session.get(url, headers=headers, timeout=12) as resp:
            if resp.status == 200:
                return await resp.text(), url
    except:
        pass
    return None, url

def smart_chunk_text(text, max_chars=3000):
    """Splits long text into chunks"""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current += p + "\n"
        else:
            chunks.append(current.strip())
            current = p + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

async def crawl_and_index(url, client_id, api_key, bot_name, bot_color, bot_avatar):
    """Crawls website & indexes content into Pinecone"""
    # Save bot config first
    save_client_config(client_id, api_key, bot_name, bot_color, bot_avatar)
    
    # Normalize URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    async with aiohttp.ClientSession() as session:
        # Step 1: Get sitemap URLs
        sitemap_urls = await fetch_sitemap(session, url)
        to_visit = set(sitemap_urls[:60]) if sitemap_urls else {url}
        visited = set()
        queue = list(to_visit)
        scraped_data = []

        # Step 2: Crawl pages (max 60 pages)
        while queue and len(visited) < 60:
            batch = queue[:8]
            queue = queue[8:]
            tasks = [fetch_url(session, u) for u in batch]
            results = await asyncio.gather(*tasks)

            for html, current_url in results:
                if not html:
                    continue
                visited.add(current_url)
                soup = BeautifulSoup(html, 'html.parser')
                
                # 🔒 Block sensitive paths
                blocked_paths = ['/login', '/admin', '/signup', '/register', '.jpg', '.png', '.pdf', '.css', '.js']
                if any(bp in current_url for bp in blocked_paths):
                    continue

                # Extract clean text
                for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript", "header"]):
                    tag.decompose()
                text = soup.get_text(separator='\n', strip=True)
                
                if len(text) > 250:  # Ignore tiny pages
                    scraped_data.append({"url": current_url, "text": text})

                # Discover internal links (if no sitemap)
                if not sitemap_urls:
                    base_domain = urlparse(url).netloc.replace("www.", "")
                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(current_url, link['href'])
                        full_url = full_url.split('#')[0].split('?')[0]  # Remove fragments & queries
                        if (base_domain in full_url 
                            and full_url not in visited 
                            and full_url not in queue
                            and not any(bp in full_url for bp in blocked_paths)):
                            queue.append(full_url)

    if not scraped_data:
        print("❌ No content scraped!")
        return False

    # Step 3: Generate embeddings & upsert to Pinecone
    try:
        genai.configure(api_key=api_key)
        vectors = []
        
        for page in scraped_data:
            chunks = smart_chunk_text(page['text'])
            for i, chunk in enumerate(chunks):
                # ✅ GET EMBEDDING (with fallback!)
                embedding = get_embedding(
                    text=chunk,
                    task_type="retrieval_document",
                    title=f"Page: {page['url']}"
                )
                vector_id = f"{client_id}_{abs(hash(page['url']))}_{i}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {"text": chunk, "url": page['url']}
                })

        # Batch upsert (50 vectors per batch)
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[i:i+batch_size], namespace=client_id)

        # Save sync timestamp
        index.upsert(vectors=[{
            "id": "config_SYNC",
            "values": [0.0] * 768,
            "metadata": {"last_sync_timestamp": int(time.time())}
        }], namespace=client_id)
        
        print(f"✅ Indexed {len(vectors)} vectors for {client_id}")
        return True
        
    except Exception as e:
        print(f"❌ Indexing Error: {e}")
        return False

def get_best_model():
    """Selects the best Gemini model"""
    return "models/gemini-1.5-flash"  # ✅ Works perfectly

# ---------------------------------
# ------------ API ---------------
# ---------------------------------

@app.get("/")
def home():
    return {"status": "FC Brain ✅ Ready! (Gemini 1.5 + text-embedding-004)"}

@app.post("/train")
async def train_bot(request: TrainRequest):
    success = await crawl_and_index(
        request.url,
        request.client_id,
        request.gemini_api_key,
        request.bot_name,
        request.bot_color,
        request.bot_avatar
    )
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Training failed")

# ✅ RESTORED ENDPOINT: /get-config  (MUST HAVE!)
@app.get("/get-config")
async def get_config(client_id: str):
    config = get_client_config(client_id)
    if not config:
        raise HTTPException(status_code=404, detail="Bot not configured. Run /train first.")
    
    return {
        "bot_name": config.get("bot_name", "AI Support"),
        "bot_color": config.get("bot_color", "#4F46E5"),
        "bot_avatar": config.get("bot_avatar", "")
    }

# ✅ RESTORED ENDPOINT: /chat
@app.post("/chat")
async def chat_bot(request: ChatRequest):
    # Get bot config (uses /get-config internally)
    config = get_client_config(request.client_id)
    if not config:
        raise HTTPException(status_code=400, detail="Bot not trained! Use /train first.")
    
    api_key = config.get("api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Invalid config: Missing API key")
    
    is_lead = check_and_save_lead(request.message, request.client_id)

    try:
        genai.configure(api_key=api_key)
        
        # ✅ GET EMBEDDING (with fallback!)
        embedding = get_embedding(
            text=request.message,
            task_type="retrieval_query"
        )
        
        # Search Pinecone
        search_results = index.query(
            namespace=request.client_id,
            vector=embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Build context for LLM
        context = "\n\n".join([
            f"SOURCE: {m['metadata']['url']}\nTEXT: {m['metadata']['text'][:1000]}" 
            for m in search_results['matches']
        ])
        
        system_prompt = f"""
You are a helpful assistant for **{request.client_id}**.
Answer ONLY using the context below. Do NOT hallucinate.

CONTEXT:
{context}

INSTRUCTIONS:
1.  Be concise.
2.  Use Markdown links like [Title](URL) when referencing a source.
3.  If you don't know the answer, say: "I don't have information about that."
"""
        if is_lead:
            system_prompt += "\n\n💡 The user provided their email. Politely acknowledge it."

        # Generate answer
        model = genai.GenerativeModel(get_best_model())
        response = model.generate_content(f"{system_prompt}\n\nUSER: {request.message}")
        bot_reply = response.text.strip()
        
        # Log chat
        log_chat(request.client_id, request.session_id, request.message, bot_reply)
        
        return {"answer": bot_reply}
    
    except Exception as e:
        print(f"❌ Chat Error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# ✅ RESTORED ENDPOINT: /get-leads
@app.post("/get-leads")
async def get_leads(request: AutoSyncRequest):
    try:
        # Dummy vector (value doesn't matter for metadata filter)
        dummy_vec = [0.01] * 768
        results = index.query(
            namespace=request.client_id,
            vector=dummy_vec,
            top_k=200,
            include_metadata=True,
            filter={"type": "lead"}
        )
        leads = []
        for m in results['matches']:
            leads.append({
                "email": m['metadata'].get('email'),
                "message": m['metadata'].get('context'),
                "timestamp": m['metadata'].get('timestamp')
            })
        return {"leads": leads}
    except Exception as e:
        print(f"❌ Get Leads Error: {e}")
        return {"leads": []}

# ✅ RESTORED ENDPOINT: /get-analytics
@app.post("/get-analytics")
async def get_analytics(request: AutoSyncRequest):
    try:
        config = get_client_config(request.client_id)
        if not config:
            return {"logs": [], "summary": "Bot not trained yet."}
        
        # Fetch chat logs
        dummy_vec = [0.01] * 768
        results = index.query(
            namespace=request.client_id,
            vector=dummy_vec,
            top_k=200,
            include_metadata=True,
            filter={"type": " chat_log"}
        )
        
        logs = []
        user_questions = []
        for m in results['matches']:
            md = m['metadata']
            logs.append({
                "session_id": md.get('session_id'),
                "user_msg": md.get('user_msg'),
                "bot_msg": md.get('bot_msg'),
                "timestamp": md.get('timestamp')
            })
            user_questions.append(md.get('user_msg'))
        
        # Generate summary if enough data
        ai_summary = "Not enough chats for analysis."
        if len(user_questions) > 10:
            try:
                genai.configure(api_key=config['api_key'])
                model = genai.GenerativeModel(get_best_model())
                prompt = (
                    f"Analyze these user questions and return TOP 3 most common topics/themes:\n"
                    f"{' | '.join(user_questions[:50])}"
                )
                summary_resp = model.generate_content(prompt)
                ai_summary = summary_resp.text.strip()
            except Exception as e:
                ai_summary = f"Analysis failed: {str(e)}"
        
        return {"logs": logs, "summary": ai_summary}
    
    except Exception as e:
        print(f"❌ Analytics Error: {e}")
        return {"logs": [], "summary": f"Error: {str(e)}"}
