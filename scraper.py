import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode
import logging
import re
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

def get_base_domain(url):
    return urlparse(url).netloc.replace('www.', '')

def clean_url(url):
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    clean_params = {k: v for k, v in query_params.items() if not k.startswith(('utm_', 'fbclid', 'gclid'))}
    new_query = urlencode(clean_params, doseq=True)
    clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), parsed.params, new_query, ''))
    return clean

def is_valid_content_page(url):
    skip_patterns = ['/login', '/cart', '/checkout', '/wp-admin', '/admin', '?s=', '/search', '/tag/', '/author/', 'cdn-cgi']
    return not any(pattern in url.lower() for pattern in skip_patterns)

def find_sitemap_urls(start_url, base_domain):
    """Hunts for the sitemap to instantly grab all website links without guessing."""
    sitemap_url = urljoin(start_url, '/sitemap.xml')
    urls = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(sitemap_url, headers=headers, timeout=5)
        if response.status_code == 200 and 'xml' in response.headers.get('Content-Type', '').lower():
            root = ET.fromstring(response.content)
            # Find all <loc> tags in the XML namespace
            for loc in root.iter():
                if 'loc' in loc.tag and loc.text:
                    href = clean_url(loc.text.strip())
                    skip_ext = ('.pdf', '.jpg', '.png', '.mp4', '.xml')
                    if get_base_domain(href) == base_domain and not any(href.lower().endswith(ext) for ext in skip_ext):
                        urls.append(href)
            logger.info(f"Sitemap Sniper found {len(urls)} URLs!")
    except Exception as e:
        logger.warning(f"No valid sitemap found at {sitemap_url}. Falling back to standard crawl.")
    return list(set(urls))

def crawl_website(start_url: str, max_pages: int = 200) -> list:
    if not start_url.startswith('http'):
        start_url = 'https://' + start_url

    start_url = clean_url(start_url)
    base_domain = get_base_domain(start_url)
    
    # 1. TRY SITEMAP FIRST
    sitemap_urls = find_sitemap_urls(start_url, base_domain)
    
    visited = set()
    to_visit = sitemap_urls if sitemap_urls else [start_url]
    extracted_data = []
    
    logger.info(f"Starting Enterprise Crawl for: {base_domain}")

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        
        if current_url in visited or not is_valid_content_page(current_url):
            continue
            
        logger.info(f"Crawling: {current_url}")
        visited.add(current_url)
        
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
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

            # 2. TARGET FOOTER/HEADER
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
                
                if href.startswith(('mailto:', 'tel:', 'javascript:')): continue
                
                href = urljoin(current_url, href)
                href = clean_url(href)
                
                if href.startswith('http') and get_base_domain(href) == base_domain:
                    skip_ext = ('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip')
                    if not any(href.lower().split('?')[0].endswith(ext) for ext in skip_ext):
                        if href not in visited and href not in to_visit and is_valid_content_page(href):
                            to_visit.append(href)
                        
                        link_entry = f"[RELEVANT SITE LINK] -> [{link_text}] URL: {href}"
                        if len(link_text) >= 2 and link_entry not in page_text_parts: 
                            page_text_parts.append(link_entry)

            # 4. GRAB MAIN BODY 
            valid_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'li', 'td', 'strong', 'em']
            for tag in soup.find_all(valid_tags):
                text = tag.get_text(separator=' ', strip=True)
                prefix = ""
                if tag.name in ['h1', 'h2']: prefix = f"[MAJOR TOPIC {tag.name.upper()}]: "
                elif tag.name in ['h3', 'h4']: prefix = f"[SUBTOPIC]: "
                
                is_contact_info = '@' in text or re.search(r'\+?\d{10}', text)
                if (len(text) > 20 or is_contact_info) and '{' not in text and 'function(' not in text:
                    formatted_text = f"{prefix}{text}"
                    if formatted_text not in page_text_parts:
                        page_text_parts.append(formatted_text)
            
            clean_text = "\n\n".join(page_text_parts)
            if len(clean_text) > 50:
                extracted_data.append({"url": current_url, "title": title, "text": clean_text})
                        
        except Exception as e:
            logger.error(f"Failed to crawl {current_url}: {str(e)}")
            
    logger.info(f"Finished crawling. Successfully extracted data from {len(extracted_data)} pages.")
    return extracted_data
