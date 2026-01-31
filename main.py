import os
import time
import re
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from fastapi import FastAPI
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

# --- DATABASE INIT ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except:
    print("Pinecone Error: Check API Key")

# --- MODELS ---
class RequestModel(BaseModel):
    url: str = ""
    client_id: str
    message: str = ""
    session_id: str = "Guest"
    gemini_api_key: str = ""
    bot_name: str = "AI Support"
    bot_color: str = "#4F46E5"
    bot_avatar: str = ""
    bot_personality: str = "Professional and concise"

# --- HELPER FUNCTIONS ---
def get_config(client_id):
    try:
        res = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        if f"config_{client_id}" in res.vectors:
            return res.vectors[f"config_{client_id}"].metadata
        return None
    except: return None

def save_lead(client_id, email, context):
    try:
        lid = f"lead_{int(time.time())}_{abs(hash(email))}"
        index.upsert(
            vectors=[{"id": lid, "values": [0.1]*768, "metadata": {"type": "lead", "email": email, "context": context, "timestamp": int(time.time())}}],
            namespace=client_id
        )
    except: pass

def log_chat(client_id, session, user, bot):
    try:
        lid = f"log_{int(time.time())}_{abs(hash(user))}"
        index.upsert(
            vectors=[{"id": lid, "values": [0.1]*768, "metadata": {"type": "chat_log", "session_id": session, "user_msg": user, "bot_msg": bot, "timestamp": int(time.time())}}],
            namespace=client_id
        )
    except: pass

# --- THE REAL CRAWLER (Restored) ---
async def fetch_url(session, url):
    # Mimic Chrome to bypass firewalls
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        async with session.get(url, headers=headers, timeout=15, ssl=False) as resp:
            if resp.status == 200: return await resp.text(), url
    except: pass
    return None, url

def smart_chunk(text):
    # Split text into chunks of ~1000 chars for better AI reading
    return [text[i:i+1000] for i in range(0, len(text), 1000)]

async def crawl_and_index(start_url, client_id, api_key):
    if not start_url.startswith('http'): start_url = 'https://' + start_url
    
    # 1. Scrape
    scraped_data = []
    async with aiohttp.ClientSession() as session:
        # We only crawl the homepage + 5 internal links for speed/stability in this tier
        html, final_url = await fetch_url(session, start_url)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract important links (Buttons, Nav)
            important_links = []
            for a in soup.find_all('a', href=True):
                full_link = urljoin(final_url, a['href'])
                text = a.get_text(strip=True)
                if text and "http" in full_link:
                    important_links.append(f"LINK: {text} -> {full_link}")
            
            # Clean Text
            for tag in soup(["script", "style", "nav", "footer"]): tag.extract()
            text = soup.get_text(separator='\n', strip=True)
            
            # Append links to text so the AI sees them
            full_content = text + "\n\n=== NAVIGABLE LINKS ===\n" + "\n".join(important_links[:20])
            scraped_data.append({"url": final_url, "text": full_content})

    # 2. Embed & Save
    genai.configure(api_key=api_key)
    vectors = []
    
    for page in scraped_data:
        chunks = smart_chunk(page['text'])
        for i, chunk in enumerate(chunks):
            try:
                emb = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
                vectors.append({
                    "id": f"{client_id}_{i}", 
                    "values": emb, 
                    "metadata": {"text": chunk, "url": page['url']}
                })
            except: pass
            
    if vectors:
        index.upsert(vectors=vectors, namespace=client_id)
        return True
    return False

# --- ENDPOINTS ---

@app.get("/")
def home(): return {"status": "FC Super-Brain Active"}

@app.post("/verify-install")
async def verify(req: RequestModel):
    target = req.url if req.url.startswith("http") else f"https://{req.url}"
    try:
        async with aiohttp.ClientSession() as session:
            html, _ = await fetch_url(session, target)
            if html and ("widget.js" in html and req.client_id in html):
                return {"status": "success", "message": "Verified"}
    except: pass
    return {"status": "failed", "message": "Script not found"}

@app.post("/train")
async def train(req: RequestModel):
    # 1. Save Config Immediately
    index.upsert(
        vectors=[{
            "id": f"config_{req.client_id}",
            "values": [1.0]*768, # Dummy vector for config
            "metadata": {
                "api_key": req.gemini_api_key,
                "bot_name": req.bot_name,
                "bot_color": req.bot_color,
                "bot_personality": req.bot_personality
            }
        }],
        namespace=req.client_id
    )
    
    # 2. Run Crawler
    success = await crawl_and_index(req.url, req.client_id, req.gemini_api_key)
    return {"status": "success" if success else "failed"}

@app.get("/get-config")
def get_conf(client_id: str):
    c = get_config(client_id)
    if c: return c
    return {"bot_name": "AI Support", "bot_color": "#4F46E5", "bot_personality": "Professional"}

@app.post("/chat")
async def chat(req: RequestModel):
    conf = get_config(req.client_id)
    if not conf: return {"answer": "Bot not configured. Please train me first."}
    
    # Lead Trap
    if re.search(r"[\w\.-]+@[\w\.-]+", req.message): 
        save_lead(req.client_id, req.message, req.message)

    try:
        genai.configure(api_key=conf['api_key'])
        
        # 1. Search Knowledge Base
        emb = genai.embed_content(model="models/text-embedding-004", content=req.message)['embedding']
        res = index.query(namespace=req.client_id, vector=emb, top_k=4, include_metadata=True)
        
        knowledge = "\n".join([m['metadata']['text'] for m in res['matches'] if 'text' in m['metadata']])
        
        # 2. Senior System Prompt
        system_prompt = f"""
        You are {conf.get('bot_name')}, a senior expert for this company.
        YOUR PERSONALITY: {conf.get('bot_personality')}.
        
        KNOWLEDGE BASE:
        {knowledge}
        
        STRICT RULES:
        1. Answer ONLY based on the Knowledge Base. 
        2. If the user asks for a page (e.g., Audit, Contact), LOOK at the 'NAVIGABLE LINKS' section in the context.
        3. ALWAYS format links as Markdown: [Click Here](https://example.com/audit).
        4. Never say "I don't have that text". If you don't know, say: "I can't find that specific page right now, would you like to speak to a human?"
        5. Be concise, professional, and helpful.
        """
        
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        ans = model.generate_content(f"{system_prompt}\n\nUser Query: {req.message}").text
        
        log_chat(req.client_id, req.session_id, req.message, ans)
        return {"answer": ans}
        
    except Exception as e: return {"answer": "I'm having trouble connecting to my brain. Please try again."}

@app.post("/get-stats")
def get_stats(req: RequestModel):
    # Honest Stats
    dummy = [0.1]*768
    leads = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "lead"})
    chats = index.query(namespace=req.client_id, vector=dummy, top_k=1000, filter={"type": "chat_log"})
    sessions = set([m['metadata']['session_id'] for m in chats['matches'] if 'session_id' in m['metadata']])
    return {"visitors": 0, "chats": len(sessions), "leads": len(leads['matches'])}

@app.post("/get-leads")
def get_leads_endpoint(req: RequestModel):
    dummy = [0.1]*768
    res = index.query(namespace=req.client_id, vector=dummy, top_k=100, filter={"type": "lead"})
    data = [{"email": m['metadata'].get('email'), "message": m['metadata'].get('context'), "date": m['metadata'].get('timestamp')} for m in res['matches']]
    return {"leads": data}

@app.post("/get-analytics")
def get_analytics_endpoint(req: RequestModel):
    dummy = [0.1]*768
    res = index.query(namespace=req.client_id, vector=dummy, top_k=100, filter={"type": "chat_log"})
    data = [{"session": m['metadata'].get('session_id'), "user": m['metadata'].get('user_msg'), "bot": m['metadata'].get('bot_msg'), "time": m['metadata'].get('timestamp')} for m in res['matches']]
    return {"logs": data}
