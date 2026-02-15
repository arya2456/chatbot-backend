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
    
    # We only want to crawl internal links, not external sites
    base_domain = urlparse(start_url).netloc

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        
        # Clean the URL to avoid crawling the same page with different # anchors
        current_url = current_url.split('#')[0]
        
        if current_url in visited:
            continue
            
        logger.info(f"Crawling: {current_url}")
        
        try:
            # We use a browser-like header so the client's site doesn't block us!
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            response = requests.get(current_url, headers=headers, timeout=10)
            
            # Skip if it's not a successful webpage
            if response.status_code != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
                visited.add(current_url)
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # REMOVE JUNK: Destroy headers, footers, scripts, and styling
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.extract()
                
            # Extract the pure, readable text
            text = soup.get_text(separator=' ', strip=True)
            
            # If the page actually has words on it, save it
            if len(text) > 100:
                extracted_data.append({
                    "url": current_url,
                    "text": text
                })
            
            visited.add(current_url)
            
            # FIND MORE PAGES: Look for internal links to crawl next
            for link in soup.find_all('a', href=True):
                next_url = urljoin(start_url, link['href'])
                
                # FIX: Strip out # anchors so we don't count page jumps as new pages
                next_url = next_url.split('#')[0]
                next_domain = urlparse(next_url).netloc
                
                # Only add if it's on the same website and we haven't seen it yet
                if next_domain == base_domain and next_url not in visited and next_url not in to_visit:
                    # FIX: Ignore email addresses, phone numbers, and Cloudflare cdn-cgi junk
                    if not next_url.startswith(('mailto:', 'tel:', 'javascript:')) and 'cdn-cgi' not in next_url:
                        to_visit.append(next_url)
                        
        except Exception as e:
            logger.error(f"Failed to crawl {current_url}: {str(e)}")
            visited.add(current_url)
            
    return extracted_data
