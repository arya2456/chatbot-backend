import os
import asyncio
import aiohttp
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
    gemini_api_key: str

# --- HELPER: AUTO-DETECT MODEL ---
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

# --- SMART CRAWLER ---
async def fetch_url(session, url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        async with session.get(url, headers=headers, timeout=8) as response:
            if response.status != 200: return None, url
            return await response.text(), url
    except:
        return None, url

def is_internal_link(base_domain, link_url):
    link_parts = urlparse(link_url)
    link_domain = link_parts.netloc.replace("www.", "")
    base_clean = base_domain.replace("www.", "")
    return base_clean in link_domain or link_domain == ""

async def crawl_website(start_url: str, max_pages: int = 50): # INCREASED TO 50
    if not start_url.startswith('http'): start_url = 'https://' + start_url
    base_domain = urlparse(start_url).netloc
    visited = set()
    to_visit = {start_url}
    scraped_data = []

    async with aiohttp.ClientSession() as session:
        while to_visit and len(visited) < max_pages:
            batch = list(to_visit)[:8] # Faster batching
            for u in batch: to_visit.remove(u)
            
            tasks = [fetch_url(session, url) for url in batch]
            results = await asyncio.gather(*tasks)

            for html, current_url in results:
                if not html: continue
                visited.add(current_url)
                
                soup = BeautifulSoup(html, 'html.parser')
                for script in soup(["script", "style", "nav", "footer", "iframe"]): 
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                
                if len(text) > 200:
                    scraped_data.append({"url": current_url, "text": text})

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    if is_internal_link(base_domain, full_url):
                        full_url = full_url.split('#')[0].rstrip('/')
                        if full_url not in visited and full_url not in to_visit:
                            # Filter junk files
                            if not any(x in full_url for x in ['.jpg', '.png', 'login', 'wp-admin', 'mailto', 'tel:']):
                                to_visit.add(full_url)
    return scraped_data

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Chatbot Brain is Active"}

@app.post("/train")
async def train_bot(request: TrainRequest):
    try:
        pages = await crawl_website(request.url)
        if not pages:
            return {"status": "error", "detail": "Could not read website."}
    except Exception as e:
        return {"status": "error", "detail": f"Crawling crashed: {str(e)}"}

    try:
        genai.configure(api_key=request.gemini_api_key)
        vectors = []
        for page in pages:
            text = page['text'][:4000]
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )
            vector_id = f"{request.client_id}_{abs(hash(page['url']))}"
            vectors.append({
                "id": vector_id,
                "values": result['embedding'],
                "metadata": {"text": text, "url": page['url']}
            })

        if vectors:
            index.upsert(vectors=vectors, namespace=request.client_id)
            
    except Exception as e:
         return {"status": "error", "detail": f"AI Error: {str(e)}"}

    return {"status": "success", "pages": len(pages)}

@app.post("/chat")
async def chat_bot(request: ChatRequest):
    try:
        genai.configure(api_key=request.gemini_api_key)
        
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=request.message,
            task_type="retrieval_query",
        )['embedding']

        search_results = index.query(
            namespace=request.client_id,
            vector=embedding,
            top_k=4, 
            include_metadata=True
        )

        context_parts = []
        for m in search_results['matches']:
            url = m['metadata'].get('url', '')
            text = m['metadata']['text']
            context_parts.append(f"URL: {url}\nINFO: {text}")
        
        context_str = "\n\n".join(context_parts)
        
        # --- NEW "SHORT & SMART" PERSONA ---
        system_instruction = f"""
        You are a smart, efficient assistant for {request.client_id}.
        
        STRICT GUIDELINES:
        1. BE CONCISE: Maximum 2-3 sentences. No fluff words like "We are thrilled" or "I understand".
        2. BE DATA-DRIVEN: Use the Source URL provided in the context.
        3. LINKS: If the user asks for a specific link (like "blogs"), look at the URLs in the context. If you see a URL with '/blog' or '/news', return it. 
        4. IF MISSING: If you don't have the exact link, check if you can guess it from the domain (e.g. domain.com/blog) or just say "Please check our main menu."
        
        CONTEXT:
        {context_str}
        """

        try:
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content(f"{system_instruction}\n\nUSER QUESTION: {request.message}")
            return {"answer": response.text}
        except:
            model = genai.GenerativeModel(get_best_model())
            response = model.generate_content(f"{system_instruction}\n\nUSER QUESTION: {request.message}")
            return {"answer": response.text}
        
    except Exception as e:
        return {"answer": f"System Error: {str(e)}"}
