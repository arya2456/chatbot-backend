import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
import re

logger = logging.getLogger(__name__)

def get_base_domain(url):
    """Strips www. to get the root domain for comparison."""
    netloc = urlparse(url).netloc
    return netloc.replace('www.', '')

def crawl_website(start_url: str, max_pages: int = 40) -> list:
    """
    Crawls a website starting from start_url up to max_pages.
    """
    # Ensure URL has http/https
    if not start_url.startswith('http'):
        start_url = 'https://' + start_url

    visited = set()
    to_visit = [start_url]
    extracted_data = []
    
    base_domain = get_base_domain(start_url)
    logger.info(f"Starting crawl for base domain: {base_domain}")

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        # Remove trailing slashes and hash fragments to prevent duplicate crawling
        current_url = current_url.split('#')[0].rstrip('/')
        
        if current_url in visited:
            continue
            
        logger.info(f"Crawling: {current_url}")
        visited.add(current_url)
        
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            response = requests.get(current_url, headers=headers, timeout=10)
            
            if response.status_code != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text_parts = []

            # 1. GRAB METADATA
            title = soup.title.string if soup.title else ""
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            desc = desc_tag['content'] if desc_tag else ""
            if title or desc:
                page_text_parts.append(f"PAGE TITLE: {title}. DESC: {desc}.")

            # 2. TARGET FOOTER/HEADER
            for structural_tag in soup.find_all(['footer', 'header', 'address', 'nav']):
                struct_text = structural_tag.get_text(separator=' | ', strip=True)
                struct_text = re.sub(r'\|\s*\|', '|', struct_text)
                if len(struct_text) > 10 and '{' not in struct_text:
                    page_text_parts.append(f"BUSINESS INFO/NAVIGATION: {struct_text}")
                structural_tag.decompose() 

            # 3. EXTRACT X-RAY LINKS
            for a_tag in soup.find_all('a', href=True):
                link_text = a_tag.get_text(separator=' ', strip=True)
                href = a_tag['href']
                
                # Make relative links absolute
                if href.startswith('/'):
                    href = urljoin(current_url, href)
                    
                href = href.split('#')[0].rstrip('/') # Clean the link
                
                # Check if it's an internal link
                link_domain = get_base_domain(href)
                
                if len(link_text) > 3 and href.startswith('http') and link_domain == base_domain:
                    page_text_parts.append(f"RELEVANT SITE LINK -> [{link_text}] URL: {href}")
                    
                    # Add to crawl queue if we haven't seen it
                    if href not in visited and href not in to_visit:
                        to_visit.append(href)

            # 4. GRAB MAIN BODY
            valid_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'li', 'span', 'td']
            for tag in soup.find_all(valid_tags):
                text = tag.get_text(separator=' ', strip=True)
                is_contact_info = '@' in text or re.search(r'\+?\d{10}', text)
                
                if (len(text) > 20 or is_contact_info) and '{' not in text and 'function(' not in text and 'var ' not in text:
                    if text not in page_text_parts:
                        page_text_parts.append(text)
            
            clean_text = "\n\n".join(page_text_parts)
            
            if len(clean_text) > 50:
                extracted_data.append({
                    "url": current_url,
                    "text": clean_text
                })
                        
        except Exception as e:
            logger.error(f"Failed to crawl {current_url}: {str(e)}")
            
    logger.info(f"Finished crawling. Successfully extracted data from {len(extracted_data)} pages.")
    return extracted_data
