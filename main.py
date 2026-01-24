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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2Nqmmq_MaJE7qaPCmboMMTC6gLsC8w7Ahx826mLb5a5Lx4vtfKx74zAF7iLhiZHjq3qE2W")

app = FastAPI()

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
    print(f"Server Start Error: {e}")

# --- MODELS ---
class TrainRequest(BaseModel):
    url: str
    client_id: str
    gemini_api_key: str
    bot_name: str = "AI Support"   # NEW: Default name
    bot_color: str = "#4F46E5"     # NEW: Default color

class ChatRequest(BaseModel):
    message: str
    client_id: str

class AutoSyncRequest(BaseModel):
    url: str
    client_id: str

class ConfigRequest(BaseModel):
    client_id: str

# --- HELPERS ---
def save_client_config(client_id, api_key, bot_name, bot_color):
    try:
        # We now save Name and Color into the config vector
        index.upsert(
            vectors=[{
                "id": f"config_{client_id}",
                "values": [1.0] * 768, 
                "metadata": {
                    "api_key": api_key, 
                    "type": "config",
                    "bot_name": bot_name,
                    "bot_color": bot_color
                }
            }],
            namespace=client_id
        )
    except Exception as e:
        print(f"Error saving config: {e}")

def get_client_config(client_id):
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

# --- CRAWLER (Standard) ---
async def fetch_sitemap(session, base_url):
    potential_sitemaps = [urljoin(base_url, "sitemap.xml"), urljoin(base_url, "wp-sitemap.xml")]
    found_urls = set()
    for sitemap_url in potential_sitemaps:
        try:
            async with session.get(sitemap_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    try:
                        root = ET.fromstring(content)
                        for elem in root.iter():
                            if 'loc' in elem.tag and elem.text: found_urls.add(elem.text.strip())
                        if found_urls: return list(found_urls)
                    except: pass
        except: pass
    return []

async def fetch_url(session, url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            if response.status != 200: return None, url
            return await response.text(), url
    except: return None, url

def smart_chunk_text(text, max_chars=3000):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n"
    if current_chunk: chunks.append(current_chunk)
    return chunks

async def crawl_and_index(url: str, client_id: str, api_key: str, bot_name: str, bot_color: str):
    print(f"--- STARTING SETUP FOR {client_id} ---")
    
    # SAVE CONFIG FIRST (Name, Color, Key)
    save_client_config(client_id, api_key, bot_name, bot_color)
    
    if not url.startswith('http'): url = 'https://' + url
    
    to_visit = set()
    scraped_data = []
    
    async with aiohttp.ClientSession() as session:
        sitemap_urls = await fetch_sitemap(session, url)
        if sitemap_urls: to_visit = set(sitemap_urls[:60]) 
        else: to_visit = {url}

        visited = set()
        queue = list(to_visit)
        
        while queue and len(visited) < 60:
            batch = queue[:8]
            queue = queue[8:]
            tasks = [fetch_url(session, u) for u in batch]
            results = await asyncio.gather(*tasks)

            for html, current_url in results:
                if not html: continue
                visited.add(current_url)
                soup = BeautifulSoup(html, 'html.parser')
                
                base_domain = urlparse(url).netloc.replace("www.", "")
                if not sitemap_urls:
                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(current_url, link['href']).split('#')[0]
                        if base_domain in full_url and full_url not in visited and full_url not in queue:
                             if not any(x in full_url for x in ['.jpg', '.png', 'login', 'admin']):
                                queue.append(full_url)

                for script in soup(["script", "style", "nav", "footer", "iframe", "noscript"]): script.extract()
                text = soup.get_text(separator='\n', strip=True)
                if len(text) > 200: scraped_data.append({"url": current_url, "text": text})

    if not scraped_data: return False

    try:
        genai.configure(api_key=api_key)
        vectors = []
        for page in scraped_data:
            chunks = smart_chunk_text(page['text'])
            for i, chunk in enumerate(chunks):
                result = genai.embed_content(model="models/text-embedding-004", content=chunk, task_type="retrieval_document")
                vector_id = f"{client_id}_{abs(hash(page['url']))}_{i}"
                vectors.append({"id": vector_id, "values": result['embedding'], "metadata": {"text": chunk, "url": page['url']}})

        if vectors:
            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                index.upsert(vectors=vectors[i:i+batch_size], namespace=client_id)
            index.upsert(vectors=[{"id": "config_SYNC", "values": [1.0] * 768, "metadata": {"last_sync_timestamp": int(time.time())}}], namespace=client_id)
            return True
    except Exception as e:
        print(f"Indexing Error: {e}")
        return False

def get_best_model():
    return "models/gemini-1.5-flash"

# --- API ENDPOINTS ---
@app.get("/")
def home(): return {"status": "SaaS Brain Active"}

# 1. NEW TRAIN ENDPOINT (Accepts Name & Color)
@app.post("/train")
async def train_bot(request: TrainRequest):
    success = await crawl_and_index(
        request.url, 
        request.client_id, 
        request.gemini_api_key, 
        request.bot_name, 
        request.bot_color
    )
    if success: return {"status": "success"}
    else: return {"status": "error"}

# 2. NEW CONFIG ENDPOINT (Widget calls this to get looks)
@app.get("/get-config")
async def get_config(client_id: str):
    config = get_client_config(client_id)
    if config:
        return {
            "bot_name": config.get("bot_name", "AI Support"),
            "bot_color": config.get("bot_color", "#4F46E5")
        }
    return {"bot_name": "Support", "bot_color": "#4F46E5"}

@app.post("/chat")
async def chat_bot(request: ChatRequest):
    config = get_client_config(request.client_id)
    if not config: return {"answer": "Error: Bot not configured."}
    
    api_key = config.get("api_key")
    is_lead = check_and_save_lead(request.message, request.client_id)

    try:
        genai.configure(api_key=api_key)
        embedding = genai.embed_content(model="models/text-embedding-004", content=request.message, task_type="retrieval_query")['embedding']
        search_results = index.query(namespace=request.client_id, vector=embedding, top_k=5, include_metadata=True)
        context = "\n\n".join([f"SOURCE: {m['metadata'].get('url','')}\nTEXT: {m['metadata']['text']}" for m in search_results['matches']])
        
        base_url = f"https://{request.client_id}"
        
        system = f"""
        You are a smart assistant for {request.client_id}.
        CONTEXT: {context}
        POTENTIAL LINKS:
        - Blogs: {base_url}/blog
        - Contact: {base_url}/contact
        INSTRUCTIONS:
        1. Answer strictly based on context.
        2. Use Markdown links: [Title](URL).
        3. Be concise.
        """
        if is_lead: system += "\n(User provided email. Confirm receipt.)"

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
