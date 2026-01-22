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

# YOUR PINECONE KEY
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
        async with session.get(url, headers=headers, timeout=10) as response:
            if response.status != 200: 
                print(f"Skipping {url} - Status {response.status}")
                return None, url
            return await response.text(), url
    except Exception as e:
        print(f"Failed to crawl {url}: {e}")
        return None, url

def is_internal_link(base_domain, link_url):
    """
    Checks if a link belongs to the same website, ignoring www.
    """
    link_parts = urlparse(link_url)
    link_domain = link_parts.netloc.replace("www.", "")
    base_clean = base_domain.replace("www.", "")
    
    return base_clean in link_domain or link_domain == ""

async def crawl_website(start_url: str, max_pages: int = 20):
    # normalize start url
    if not start_url.startswith('http'): start_url = 'https://' + start_url
    
    base_domain = urlparse(start_url).netloc
    visited = set()
    to_visit = {start_url}
    scraped_data = []

    print(f"--- STARTING CRAWL: {start_url} ---")

    async with aiohttp.ClientSession() as session:
        while to_visit and len(visited) < max_pages:
            # Grab a batch of URLs
            batch = list(to_visit)[:5] 
            for u in batch: to_visit.remove(u)
            
            tasks = [fetch_url(session, url) for url in batch]
            results = await asyncio.gather(*tasks)

            for html, current_url in results:
                if not html: continue
                visited.add(current_url)
                print(f"Crawled: {current_url}") # LOGGING PROOF
                
                soup = BeautifulSoup(html, 'html.parser')
                
                # Clean text
                for script in soup(["script", "style", "nav", "footer", "iframe"]): 
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                
                if len(text) > 200: # Only save pages with actual content
                    scraped_data.append({"url": current_url, "text": text})

                # Find ALL links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(current_url, href)
                    
                    # Filter for internal links only
                    if is_internal_link(base_domain, full_url):
                        # Clean anchors (#section)
                        full_url = full_url.split('#')[0].rstrip('/')
                        
                        if full_url not in visited and full_url not in to_visit:
                            # Avoid junk links
                            if not any(x in full_url for x in ['.jpg', '.png', '.pdf', 'login', 'wp-admin', 'mailto']):
                                to_visit.add(full_url)
                                
    print(f"--- CRAWL FINISHED: Found {len(scraped_data)} pages ---")
    return scraped_data

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Chatbot Brain is Active"}

@app.post("/train")
async def train_bot(request: TrainRequest):
    print(f"Received Training Request for {request.url}")
    
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

    return {"status": "success", "pages_crawled": len(pages)}

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
            top_k=5, 
            include_metadata=True
        )

        context = "\n".join([m['metadata']['text'] for m in search_results['matches']])
        
        # Auto-Select Model
        try:
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content(f"Context: {context}\n\nQuestion: {request.message}\nAnswer:")
            return {"answer": response.text}
        except:
            model = genai.GenerativeModel(get_best_model())
            response = model.generate_content(f"Context: {context}\n\nQuestion: {request.message}\nAnswer:")
            return {"answer": response.text}
        
    except Exception as e:
        return {"answer": f"System Error: {str(e)}"}
