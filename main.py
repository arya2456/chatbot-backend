import os
import asyncio
import aiohttp
import time
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import google.generativeai as genai

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = "chatbot-index"
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
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# --- MODELS ---
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


# --- HELPERS ---
def save_client_config(client_id, api_key, bot_name, bot_color, bot_avatar):
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
    except Exception as e:
        print(f"Config Error: {e}")


def get_client_config(client_id):
    try:
        response = index.fetch(ids=[f"config_{client_id}"], namespace=client_id)
        return response.vectors[f"config_{client_id}"].metadata if f"config_{client_id}" in response.vectors else None
    except:
        return None


def check_and_save_lead(message, client_id):
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
                "metadata": {"type": "lead", "email": email, "context": message, "timestamp": timestamp}
            }],
            namespace=client_id
        )
        return True
    except:
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


# --- EMBEDDING HELPER (UPDATED) ---
def get_embedding(text: str, task_type: str = "retrieval_document", title: str = None):
    """Uses latest Google embedding model"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type=task_type,
        title=title
    )
    return result['embedding']


# --- CRAWLER LOGIC ---
async def fetch_sitemap(session, base_url):
    potential_sitemaps = [urljoin(base_url, "sitemap.xml"), urljoin(base_url, "wp-sitemap.xml")]
    for sitemap_url in potential_sitemaps:
        try:
            async with session.get(sitemap_url, timeout=10) as resp:
                if resp.status == 200:
                    root = ET.fromstring(await resp.text())
                    urls = {elem.text.strip() for elem in root.iter() if 'loc' in elem.tag and elem.text}
                    if urls:
                        return list(urls)
        except:
            continue
    return []


async def fetch_url(session, url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FC-Bot/1.0)"}
    try:
        async with session.get(url, headers=headers, timeout=12) as resp:
            if resp.status == 200:
                return await resp.text(), url
    except:
        pass
    return None, url


def smart_chunk_text(text, max_chars=3000):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current += para + "\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks


async def crawl_and_index(url, client_id, api_key, bot_name, bot_color, bot_avatar):
    save_client_config(client_id, api_key, bot_name, bot_color, bot_avatar)

    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

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
                if not html:
                    continue
                visited.add(current_url)
                soup = BeautifulSoup(html, 'html.parser')

                # Smart internal crawling if no sitemap
                if not sitemap_urls:
                    base_domain = urlparse(url).netloc.replace("www.", "")
                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(current_url, link['href']).split('#')[0].split('?')[0]
                        if (base_domain in full_url and full_url not in visited and full_url not in queue
                                and not any(x in full_url for x in ['.jpg', '.png', '.pdf', 'login', 'admin', 'signup'])):
                            queue.append(full_url)

                # Clean text
                for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript", "header"]):
                    tag.extract()
                text = soup.get_text(separator='\n', strip=True)

                if len(text) > 250:
                    scraped_data.append({"url": current_url, "text": text})

    if not scraped_data:
        return False

    try:
        genai.configure(api_key=api_key)
        vectors = []

        for page in scraped_data:
            chunks = smart_chunk_text(page['text'])
            for i, chunk in enumerate(chunks):
                embedding = get_embedding(
                    text=chunk,
                    task_type="retrieval_document",
                    title=f"Page from {page['url']}"   # Improves retrieval quality
                )
                vector_id = f"{client_id}_{abs(hash(page['url']))}_{i}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {"text": chunk, "url": page['url']}
                })

        # Batch upsert
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[i:i + batch_size], namespace=client_id)

        # Save last sync time
        index.upsert(
            vectors=[{
                "id": "config_SYNC",
                "values": [0.0] * 768,
                "metadata": {"last_sync_timestamp": int(time.time())}
            }],
            namespace=client_id
        )
        return True

    except Exception as e:
        print(f"Indexing Error: {e}")
        return False


def get_best_model():
    return "models/gemini-1.5-flash"


# --- API ENDPOINTS ---
@app.get("/")
def home():
    return {"status": "FC Brain - Running (Gemini 1.5 + text-embedding-004)"}


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
    return {"status": "success" if success else "failed"}


@app.post("/chat")
async def chat_bot(request: ChatRequest):
    config = get_client_config(request.client_id)
    if not config:
        return {"answer": "Error: Bot not trained yet."}

    api_key = config.get("api_key")
    is_lead = check_and_save_lead(request.message, request.client_id)

    try:
        genai.configure(api_key=api_key)

        # Updated Embedding Call
        embedding = get_embedding(
            text=request.message,
            task_type="retrieval_query"
        )

        search_results = index.query(
            namespace=request.client_id,
            vector=embedding,
            top_k=5,
            include_metadata=True
        )

        context = "\n\n".join([
            f"SOURCE: {m['metadata'].get('url','')}\nTEXT: {m['metadata']['text']}"
            for m in search_results['matches']
        ])

        system_prompt = f"""
You are a helpful assistant for {request.client_id}.
Use only the following context to answer. Be concise and accurate.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer only using the context above
- Use Markdown links when possible: [Title](URL)
- If unsure, say "I don't have that information."
"""

        if is_lead:
            system_prompt += "\nThe user just shared their email. Acknowledge it politely."

        model = genai.GenerativeModel(get_best_model())
        response = model.generate_content(f"{system_prompt}\n\nUSER: {request.message}")

        bot_reply = response.text.strip()
        log_chat(request.client_id, request.session_id, request.message, bot_reply)

        return {"answer": bot_reply}

    except Exception as e:
        print(f"Chat Error: {e}")
        return {"answer": "Sorry, something went wrong on our end."}


# Keep your other endpoints (/get-leads, /get-analytics, /get-config) as they are...
# (I left them unchanged since they were already working)
