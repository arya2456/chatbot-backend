import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

logger = logging.getLogger(__name__)

def crawl_website(start_url: str, max_pages: int = 15) -> list:
    """
    Crawls a website starting from start_url up to max_pages.
    Returns a list of dictionaries: [{"url": "...", "text": "..."}, ...]
    """
    visited = set()
    to_visit = [start_url]
    extracted_data = []
    
    base_domain = urlparse(start_url).netloc

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        current_url = current_url.split('#')[0]
        
        if current_url in visited:
            continue
            
        logger.info(f"Crawling: {current_url}")
        
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            response = requests.get(current_url, headers=headers, timeout=10)
            
            if response.status_code != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
                visited.add(current_url)
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # --- THE SNIPER FIX ---
            # Instead of grabbing everything, ONLY grab actual readable text tags
            valid_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'li', 'span']
            page_text_parts = []
            
            for tag in soup.find_all(valid_tags):
                text = tag.get_text(separator=' ', strip=True)
                # Filter out tiny words and curly braces (which usually means it accidentally grabbed CSS/JS code)
                if len(text) > 20 and '{' not in text and 'function(' not in text:
                    page_text_parts.append(text)
            
            # Join all the clean paragraphs together
            clean_text = " ".join(page_text_parts)
            
            # If we found actual content, save it!
            if len(clean_text) > 100:
                extracted_data.append({
                    "url": current_url,
                    "text": clean_text
                })
            
            visited.add(current_url)
            
            for link in soup.find_all('a', href=True):
                next_url = urljoin(start_url, link['href']).split('#')[0]
                next_domain = urlparse(next_url).netloc
                
                if next_domain == base_domain and next_url not in visited and next_url not in to_visit:
                    if not next_url.startswith(('mailto:', 'tel:', 'javascript:')) and 'cdn-cgi' not in next_url:
                        to_visit.append(next_url)
                        
        except Exception as e:
            logger.error(f"Failed to crawl {current_url}: {str(e)}")
            visited.add(current_url)
            
    return extracted_data
