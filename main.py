import os
import time
import asyncio
import aiohttp
import logging
import uuid
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not PINECONE_API_KEY:
    logger.error("CRITICAL: PINECONE_API_KEY is missing.")

app = FastAPI(title="Omni-Brain v18.0 (Direct Injection)", version="18.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def connect_db():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index(PINECONE_INDEX_NAME)
    except: return None

index = connect_db()

# --- 2. DATA MODELS (Updated to accept Key) ---
class ChatRequest(BaseModel):
    message: str
    client_id: str
    session_id: str = "Guest"
    api_key: str = "" # <--- NEW FIELD: The Mini-Brain Key

class TrainRequest(BaseModel):
    client_id: str
    url: str
    gemini_api_key: str 
    bot_name: str = "AI Assistant"
    bot_personality: str = "Professional"

# --- 3. MODEL HELPERS ---
def get_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash") # Stable Model

def safe_embed(text, api_key):
    genai.configure(api_key=api_key)
    try:
        return genai.embed_content(model="models/embedding-001", content=text)['embedding']
    except:
        return [0.0] * 768

# --- 4. SCRAPER ---
async def deep_scraper_engine(start_url: str, client_id: str, api_key: str):
    if not start_url.startswith("http"): start_url = f"https://{start_url}"
    domain = urlparse(start_url).netloc
    visited, vectors, seen_hashes = set(), [], set()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))

    async with aiohttp.ClientSession() as session:
        while not queue.empty() and len(visited) < 30:
            url, depth = await queue.get()
            if url in visited or depth > 3: continue
            visited.add(url)
            try:
                async with session.get(url, timeout=10, ssl=False) as resp:
                    if resp.status != 200: continue
                    soup = BeautifulSoup(await resp.text(), 'html.parser')
                    for a in soup.find_all('a', href=True):
                        link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                        if urlparse(link).netloc == domain and link not in visited:
                            await queue.put((link, depth + 1))
                    for x in soup(['script', 'style', 'nav', 'footer', 'aside']): x.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    if len(text) < 200: continue

                    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                    for i, chunk in enumerate(chunks):
                        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                        if h in seen_hashes: continue
                        seen_hashes.add(h)
                        
                        # Use the key passed directly from Dashboard
                        emb = safe_embed(chunk, api_key)
                        vectors.append({
                            "id": f"neural_{uuid.uuid4()}", "values": emb,
                            "metadata": {"text": chunk, "url": url, "type": "knowledge"} 
                        })
                    if len(vectors) > 20:
                        index.upsert(vectors=vectors, namespace=client_id)
                        vectors = []
            except: continue
        if vectors: index.upsert(vectors=vectors, namespace=client_id)

# --- 5. CHAT ENGINE (DIRECT INJECTION) ---
@app.post("/chat")
async def saas_brain_chat(req: ChatRequest):
    try:
        # 1. CHECK FOR DIRECT KEY (The "Mini Brain" Logic)
        active_key = req.api_key.strip()
        
        # Fallback: If widget didn't send key, try DB (Old method)
        if not active_key:
            try: 
                res = index.fetch(ids=[f"config_{req.client_id}"], namespace=req.client_id)
                if res.vectors: active_key = res.vectors[f"config_{req.client_id}"].metadata.get("api_key")
            except: pass
            
        if not active_key:
            return {"answer": "System Error: No API Key found in Widget or Database."}

        # 2. FETCH KNOWLEDGE
        emb = safe_embed(req.message, active_key)
        search = index.query(namespace=req.client_id, vector=emb, top_k=4, include_metadata=True, filter={"type": "knowledge"})
        
        context_str = "\n".join([f"INFO: {m['metadata']['text']}" for m in search['matches']])

        # 3. GENERATE
        model = get_model(active_key)
        
        sys_msg = f"""
        You are an AI assistant. Tone: Professional.
        Answer strictly based on this knowledge:
        {context_str}
        """
        
        try:
            ans = model.generate_content(f"{sys_msg}\n\nUSER: {req.message}").text
        except Exception as e:
            return {"answer": f"Google API Error: {str(e)}"}

        return {"answer": ans}

    except Exception as e:
        return {"answer": f"Critical Error: {str(e)}"}

# --- 6. TRAIN ---
@app.post("/train")
async def train_saas_engine(req: TrainRequest, bg: BackgroundTasks):
    # Save config to DB just in case, but rely on Direct Injection
    meta = {"type": "config", "api_key": req.gemini_api_key, "bot_name": req.bot_name}
    index.upsert(vectors=[{"id": f"config_{req.client_id}", "values": [1.0]*768, "metadata": meta}], namespace=req.client_id)
    
    bg.add_task(deep_scraper_engine, req.url, req.client_id, req.gemini_api_key)
    return {"status": "success", "message": "Training Started."}

# --- 7. UTILS ---
@app.post("/verify-install")
async def verify_engine(req: BaseModel): return {"status": "success"}
@app.get("/")
def health(): return {"status": "Omni-Brain v18.0 Active"}
