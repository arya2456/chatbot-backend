import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
import re

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
            page_text_parts = []

            # 1. GRAB VITAL METADATA (Who are they?)
            if current_url == start_url:
                title = soup.title.string if soup.title else ""
                desc_tag = soup.find('meta', attrs={'name': 'description'})
                desc = desc_tag['content'] if desc_tag else ""
                if title or desc:
                    page_text_parts.append(f"COMPANY OVERVIEW: {title}. {desc}.")

            # 2. TARGET FOOTER, HEADER, & ADDRESS EXPLICITLY (Contact Info)
            for structural_tag in soup.find_all(['footer', 'header', 'address', 'nav']):
                struct_text = structural_tag.get_text(separator=' | ', strip=True)
                # Clean up the pipe separators
                struct_text = re.sub(r'\|\s*\|', '|', struct_text)
                if len(struct_text) > 10 and '{' not in struct_text:
                    page_text_parts.append(f"BUSINESS INFO/NAVIGATION: {struct_text}")
                # Remove these tags so we don't duplicate them in the next step
                structural_tag.decompose() 

            # 3. GRAB THE MAIN BODY CONTENT
            valid_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'li', 'span', 'td', 'a']
            
            for tag in soup.find_all(valid_tags):
                text = tag.get_text(separator=' ', strip=True)
                
                # We want to keep it if it's long enough, OR if it looks like an email/phone number
                is_contact_info = '@' in text or re.search(r'\+?\d{10}', text)
                
                if (len(text) > 20 or is_contact_info) and '{' not in text and 'function(' not in text and 'var ' not in text:
                    # Don't add if it's already in the list to prevent weird looping repeats
                    if text not in page_text_parts:
                        page_text_parts.append(text)
            
            # Join everything logically
            clean_text = "\n\n".join(page_text_parts)
            
            # If we found actual content, save it!
            if len(clean_text) > 50:
                extracted_data.append({
                    "url": current_url,
                    "text": clean_text
                })
            
            visited.add(current_url)
            
            # Find more links to crawl
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
