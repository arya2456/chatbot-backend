import os
import asyncio
import aiohttp
import time
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = "chatbot-index"
# NOTE: We use os.getenv to keep it safe, but fallback to your provided key if env not set
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE CONNECTION ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Server Start Error: {e}")

# --- DATA MODELS ---
class TrainRequest(BaseModel):
    url: str
    client_id: str
    gemini_api_key: str

class ChatRequest(BaseModel):
    message: str
    client_id: str

class AutoSyncRequest(BaseModel):
    url: str
    client_id: str

# --- HELPER: KEY MANAGEMENT ---
def save_client_key(client_id, api_key):
    try:
        index.upsert(
            vectors=[{
                "id": f"config_{client_id}",
                "values": [1.0] * 768, # Non-zero vector for Pinecone
                "metadata": {"api_key": api_key, "type": "config"}
            }],
            namespace=client_id
        )
    except Exception as e:
        print(f"Error saving key: {e}")

def get_client_key(client_id):
    try:
        response = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if f"config_{client_id}" in response.vectors:
            return response.vectors[f"config_{client_id}"].metadata.get("api_key")
        return None
    except:
        return None

# --- HELPER: LEAD CAPTURE ---
def check_and_save_lead(message, client_id):
    """
    Scans message for email addresses. If found, saves as a lead.
    """
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_regex, message)
    if match:
        email = match.group()
        print(f"Lead detected: {email}")
        timestamp = int(time.time())
        # We store leads as a special vector in Pinecone
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
            return True
        except Exception as e:
            print(f"Lead save error: {e}")
    return False

# --- FEATURE: SITEMAP SCANNER (A-Z Coverage) ---
async def fetch_sitemap(session, base_url):
    """
    Tries to find sitemap.xml to get ALL pages.
    """
    potential_sitemaps = [
        urljoin(base_url, "sitemap.xml"),
        urljoin(base_url, "sitemap_index.xml"),
        urljoin(base_url, "wp-sitemap.xml")
    ]
    
    found_urls = set()
    
    for sitemap_url in potential_sitemaps:
        try:
            async with session.get(sitemap_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    try:
                        root = ET.fromstring(content)
                        # Extract all <loc> tags
                        for elem in root.iter():
                            if 'loc' in elem.tag and elem.text:
                                found_urls.add(elem.text.strip())
                        if found_urls:
                            print(f"Sitemap found at {sitemap_url}: {len(found_urls)} URLs")
                            return list(found_urls)
                    except:
                        pass # XML parse failed, try next
        except:
            pass
    return []

# --- CRAWLER LOGIC ---
async def fetch_url(session, url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            if response.status != 200: return None, url
            return await response.text(), url
    except:
        return None, url

def smart_chunk_text(text, max_chars=3000):
    """
    Intelligently splits text by paragraphs instead of random cuts.
    """
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

async def crawl_and_index(url: str, client_id: str, api_key: str):
    print(f"--- STARTING ULTRA-SYNC FOR {client_id} ---")
    save_client_key(client_id, api_key)
    
    if not url.startswith('http'): url = 'https://' + url
    
    to_visit = set()
    scraped_data = []
    
    async with aiohttp.ClientSession() as session:
        # 1. Try Sitemap First (The "A-Z" Method)
        sitemap_urls = await fetch_sitemap(session, url)
        if sitemap_urls:
            # Limit to 60 pages to save time/resources
            to_visit = set(sitemap_urls[:60]) 
        else:
            # Fallback to Homepage
            to_visit = {url}

        # 2. Crawler Loop
        visited = set()
        queue = list(to_visit)
        
        # Process in batches
        while queue and len(visited) < 60:
            batch = queue[:8]
            queue = queue[8:]
            
            tasks = [fetch_url(session, u) for u in batch]
            results = await asyncio.gather(*tasks)

            for html, current_url in results:
                if not html: continue
                visited.add(current_url)
                
                soup = BeautifulSoup(html, 'html.parser')
                for script in soup(["script", "style", "nav", "footer", "iframe", "noscript"]): 
                    script.extract()
                text = soup.get_text(separator='\n', strip=True)
                
                if len(text) > 200:
                    scraped_data.append({"url": current_url, "text": text})

                # If we didn't have a sitemap, find links manually
                if not sitemap_urls:
                    base_domain = urlparse(url).netloc.replace("www.", "")
                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(current_url, link['href']).split('#')[0]
                        if base_domain in full_url and full_url not in visited and full_url not in queue:
                             if not any(x in full_url for x in ['.jpg', '.png', 'login', 'admin']):
                                queue.append(full_url)

    if not scraped_data: return False

    # 3. Embedding & Indexing
    try:
        genai.configure(api_key=api_key)
        vectors = []
        
        for page in scraped_data:
            # Use Smart Chunking
            chunks = smart_chunk_text(page['text'])
            
            for i, chunk in enumerate(chunks):
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=chunk,
                    task_type="retrieval_document",
                )
                # Create unique ID for every chunk
                vector_id = f"{client_id}_{abs(hash(page['url']))}_{i}"
                vectors.append({
                    "id": vector_id,
                    "values": result['embedding'],
                    "metadata": {"text": chunk, "url": page['url']}
                })

        if vectors:
            # Batch upsert (Pinecone limit is 100 per request usually)
            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                index.upsert(vectors=vectors[i:i+batch_size], namespace=client_id)
            
            # Save Sync Timestamp
            index.upsert(
                vectors=[{
                    "id": "config_SYNC",
                    "values": [1.0] * 768, 
                    "metadata": {"last_sync_timestamp": int(time.time())}
                }],
                namespace=client_id
            )
            return True
            
    except Exception as e:
        print(f"Indexing Error: {e}")
        return False

# --- AUTO-DETECT MODEL ---
def get_best_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name: return m.name
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                return m.name
    except:
        return "models/gemini-1.5-flash"
    return "models/gemini-pro"

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Ultra-Intelligent Brain Active", "features": ["Sitemap", "Leads", "Smart-Chunking"]}

@app.post("/train")
async def train_bot(request: TrainRequest):
    success = await crawl_and_index(request.url, request.client_id, request.gemini_api_key)
    if success: return {"status": "success"}
    else: return {"status": "error"}

@app.post("/trigger-sync")
async def trigger_sync(request: AutoSyncRequest, background_tasks: BackgroundTasks):
    api_key = get_client_key(request.client_id)
    if not api_key: return {"status": "No Key Found"}
    
    # Check Time
    fetch_response = index.fetch(ids=["config_SYNC"], namespace=request.client_id)
    should_sync = False
    current_time = int(time.time())
    
    if "config_SYNC" in fetch_response.vectors:
        last_sync = int(fetch_response.vectors["config_SYNC"].metadata.get("last_sync_timestamp", 0))
        if (current_time - last_sync) > 86400: # 24 Hours
            should_sync = True
    else:
        should_sync = True
    
    if should_sync:
        background_tasks.add_task(crawl_and_index, request.url, request.client_id, api_key)
        return {"status": "Sync Started"}
    return {"status": "Up to date"}

@app.post("/chat")
async def chat_bot(request: ChatRequest):
    api_key = get_client_key(request.client_id)
    if not api_key: return {"answer": "Security Error: Please re-train bot."}

    # 1. LEAD DETECTION
    is_lead = check_and_save_lead(request.message, request.client_id)

    try:
        genai.configure(api_key=api_key)
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=request.message,
            task_type="retrieval_query",
        )['embedding']

        search_results = index.query(
            namespace=request.client_id, 
            vector=embedding, 
            top_k=5, 
            include_metadata=True
        )

        context = "\n\n".join([f"SOURCE: {m['metadata'].get('url','')}\nTEXT: {m['metadata']['text']}" for m in search_results['matches']])
        
        # --- SALES AGENT PERSONA ---
        system = f"""
        You are a smart Sales Assistant for {request.client_id}.
        
        YOUR GOAL: Answer questions accurately and encourage the user to provide their email for more info.
        
        CONTEXT FROM WEBSITE:
        {context}
        
        INSTRUCTIONS:
        1. Answer based ONLY on the context.
        2. If the user asks for a price, audit, or service, say: "I can help with that! What's your email so I can send the details?"
        3. Keep answers short (under 3 sentences).
        4. If you provided a link, format it like: [Link Text](URL).
        """
        
        if is_lead:
            system += "\nNOTE: The user just provided their email. Thank them and say someone will contact them soon."

        try:
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            res = model.generate_content(f"{system}\n\nUSER: {request.message}")
            return {"answer": res.text}
        except:
            model = genai.GenerativeModel(get_best_model())
            res = model.generate_content(f"{system}\n\nUSER: {request.message}")
            return {"answer": res.text}

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

# --- ENDPOINT TO VIEW LEADS (For Dashboard) ---
@app.post("/get-leads")
async def get_leads(request: AutoSyncRequest):
    # This retrieves the last 100 leads saved in Pinecone
    # Note: Vector search isn't ideal for "List all", but we used a metadata filter trick
    # Actually, Pinecone doesn't support "List All" easily. 
    # For now, we return a placeholder or implement a specialized fetch if needed.
    # A true 'List Leads' usually requires a SQL DB. 
    # For this No-SQL setup, we'd query for the "type": "lead" metadata.
    
    # Simple hack: Query for the word "mail" which appears in all emails
    try:
        genai.configure(api_key=get_client_key(request.client_id))
        dummy_embed = genai.embed_content(model="models/text-embedding-004", content="mail", task_type="retrieval_query")['embedding']
        
        results = index.query(
            namespace=request.client_id,
            vector=dummy_embed,
            top_k=20,
            include_metadata=True,
            filter={"type": "lead"}
        )
        
        leads = []
        for m in results['matches']:
            leads.append({
                "email": m['metadata'].get('email'),
                "message": m['metadata'].get('context'),
                "date": m['metadata'].get('timestamp')
            })
        return {"leads": leads}
    except:
        return {"leads": []}
