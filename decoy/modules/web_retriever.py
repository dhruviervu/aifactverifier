import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
from urllib.parse import urlparse
import re
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebRetriever:
    def __init__(self):
        """Initialize web retriever with required models"""
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and from allowed domains"""
        try:
            result = urlparse(url)
            # List of trusted domains
            trusted_domains = ['wikipedia.org', 'reuters.com', 'apnews.com', 
                             'bbc.com', 'nytimes.com']
            return any(domain in result.netloc for domain in trusted_domains)
        except:
            return False
            
    def clean_text(self, text: str) -> str:
        """Clean web scraped text"""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove scripts and styles
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
        
        # Remove other HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
        
    def scrape_url(self, url: str) -> str:
        """Scrape text content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['nav', 'header', 'footer', 'aside']):
                element.decompose()
                
            # Get main content
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            
            if main_content:
                text = main_content.get_text()
                return self.clean_text(text)
            return ""
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return ""
            
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search web for relevant pages
        Returns list of {url, title, snippet} dicts
        """
        # NOTE: This is a placeholder. In production, use a real search API
        # like Google Custom Search API or Bing Web Search API
        
        example_results = [
            {
                'url': 'https://www.reuters.com/example1',
                'title': 'Example Article 1',
                'snippet': 'Relevant snippet about ' + query
            },
            {
                'url': 'https://www.bbc.com/example2',
                'title': 'Example Article 2', 
                'snippet': 'Another relevant snippet about ' + query
            }
        ]
        
        return example_results[:num_results]
        
    def rank_passages(self, query: str, passages: List[str]) -> List[Dict]:
        """
        Rank passages by relevance using semantic similarity
        Returns list of {passage, score} dicts
        """
        if not passages:
            return []
            
        # Encode query and passages
        query_embedding = self.encoder.encode([query])[0]
        passage_embeddings = self.encoder.encode(passages)
        
        # Compute similarities
        similarities = [np.dot(query_embedding, p_emb)/(np.linalg.norm(query_embedding)*np.linalg.norm(p_emb))
                       for p_emb in passage_embeddings]
                       
        # Create ranked list
        ranked = [
            {
                'passage': passage,
                'score': float(score)
            }
            for passage, score in zip(passages, similarities)
        ]
        
        # Sort by score
        ranked.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked
        
    def retrieve_evidence(self, claim: str) -> Dict:
        """
        Main evidence retrieval function that:
        1. Searches web
        2. Scrapes content
        3. Ranks passages
        Returns dict with best evidence
        """
        # Search web
        search_results = self.search_web(claim)
        
        all_passages = []
        sources = []
        
        # Scrape each result
        for result in search_results:
            if self.is_valid_url(result['url']):
                content = self.scrape_url(result['url'])
                if content:
                    all_passages.append(content)
                    sources.append(result['url'])
                    
        if not all_passages:
            return {
                "evidence": "",
                "source": "",
                "score": 0.0
            }
            
        # Rank passages
        ranked = self.rank_passages(claim, all_passages)
        
        if not ranked:
            return {
                "evidence": "",
                "source": "",
                "score": 0.0
            }
            
        # Get best evidence
        best_evidence = ranked[0]
        source = sources[all_passages.index(best_evidence['passage'])]
        
        return {
            "evidence": best_evidence['passage'],
            "source": source,
            "score": best_evidence['score']
        }