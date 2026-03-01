import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode
import logging
import re

logger = logging.getLogger(__name__)

def get_base_domain(url):
    """Strips www. to get the root domain for comparison."""
    return urlparse(url).netloc.replace('www.', '')

def clean_url(url):
    """Removes fragments, trailing slashes, and useless tracking parameters."""
    parsed = urlparse(url)
    
    # Remove tracking queries but keep important ones (like ?page=2)
    query_params = parse_qs(parsed.query)
    clean_params = {k: v for k, v in query_params.items() if not k.startswith(('utm_', 'fbclid', 'gclid'))}
    new_query = urlencode(clean_params, doseq=True)
    
    # Rebuild URL without fragment and with cleaned query
    clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), parsed.params, new_query, ''))
    return clean

def is_valid_content_page(url):
    """Skips obvious non-content pages to save crawl time and DB space."""
    skip_patterns = ['/login', '/cart', '/checkout', '/wp-admin', '/admin', '?s=', '/search', '/tag/', '/author/']
    return not any(pattern in url.lower() for pattern in skip_patterns)

def crawl_website(start_url: str, max_pages: int = 40) -> list:
    if not start_url.startswith('http'):
        start_url = 'https://' + start_url

    start_url = clean_url(start_url)
    visited = set()
    to_visit = [start_url]
    extracted_data = []
    
    base_domain = get_base_domain(start_url)
    logger.info(f"Starting Pro Crawl for: {base_domain}")

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        
        if current_url in visited or not is_valid_content_page(current_url):
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
            title = soup.title.string.strip() if soup.title else "Untitled Page"
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            desc = desc_tag['content'].strip() if desc_tag else ""
            if title or desc:
                page_text_parts.append(f"[PAGE META] TITLE: {title} | DESC: {desc}")

            # 2. TARGET FOOTER/HEADER (Contact Info)
            for structural_tag in soup.find_all(['footer', 'header', 'address', 'nav']):
                struct_text = structural_tag.get_text(separator=' | ', strip=True)
                struct_text = re.sub(r'\|\s*\|', '|', struct_text)
                if len(struct_text) > 10 and '{' not in struct_text:
                    page_text_parts.append(f"[GLOBAL INFO]: {struct_text}")
                structural_tag.decompose() 

            # 3. EXTRACT X-RAY LINKS
            for a_tag in soup.find_all('a', href=True):
                link_text = a_tag.get_text(separator=' ', strip=True)
                href = a_tag['href']
                
                if href.startswith('/'):
                    href = urljoin(current_url, href)
                    
                href = clean_url(href)
                link_domain = get_base_domain(href)
                
                if len(link_text) > 3 and href.startswith('http') and link_domain == base_domain:
                    page_text_parts.append(f"[RELEVANT SITE LINK] -> [{link_text}] URL: {href}")
                    
                    if href not in visited and href not in to_visit and is_valid_content_page(href):
                        to_visit.append(href)

            # 4. GRAB MAIN BODY (With Structural Tagging)
            valid_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'li', 'td', 'strong', 'em']
            for tag in soup.find_all(valid_tags):
                text = tag.get_text(separator=' ', strip=True)
                
                # Tag Headers so AI knows they are important
                prefix = ""
                if tag.name in ['h1', 'h2']:
                    prefix = f"[MAJOR TOPIC {tag.name.upper()}]: "
                elif tag.name in ['h3', 'h4']:
                    prefix = f"[SUBTOPIC]: "
                
                is_contact_info = '@' in text or re.search(r'\+?\d{10}', text)
                
                if (len(text) > 20 or is_contact_info) and '{' not in text and 'function(' not in text and 'var ' not in text:
                    formatted_text = f"{prefix}{text}"
                    if formatted_text not in page_text_parts:
                        page_text_parts.append(formatted_text)
            
            clean_text = "\n\n".join(page_text_parts)
            
            if len(clean_text) > 50:
                extracted_data.append({
                    "url": current_url,
                    "title": title, # Pass title to main.py
                    "text": clean_text
                })
                        
        except Exception as e:
            logger.error(f"Failed to crawl {current_url}: {str(e)}")
            
    logger.info(f"Finished crawling. Successfully extracted data from {len(extracted_data)} pages.")
    return extracted_data
