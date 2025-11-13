import wikipediaapi
import spacy
from nltk.corpus import wordnet
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikiRetriever:
    def __init__(self):
        """Initialize Wikipedia retriever with NLP models and APIs"""
        self.wiki = wikipediaapi.Wikipedia('en')
        self.nlp = spacy.load('en_core_web_lg')
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
    def get_entity_variations(self, text: str) -> List[str]:
        """Get variations of named entities using spaCy and WordNet"""
        doc = self.nlp(text)
        variations = set()
        
        # Add named entities
        for ent in doc.ents:
            variations.add(ent.text)
            # Add lemmatized version
            variations.add(ent.lemma_)
        
        # Add WordNet synonyms for entities
        for token in doc:
            if token.ent_type_:
                synsets = wordnet.synsets(token.text)
                for syn in synsets:
                    variations.update(syn.lemma_names())
                    
        return list(variations)

    def get_wiki_text(self, query: str, fallback_queries: Optional[List[str]] = None) -> Tuple[str, str]:
        """
        Get Wikipedia text for a query with fallback options
        Returns: (text, source_page)
        """
        # Try primary query
        page = self.wiki.page(query)
        if page.exists():
            return page.text, page.title
            
        # Try variations
        variations = self.get_entity_variations(query)
        for var in variations:
            page = self.wiki.page(var)
            if page.exists():
                return page.text, page.title
                
        # Try fallback queries
        if fallback_queries:
            for q in fallback_queries:
                page = self.wiki.page(q)
                if page.exists():
                    return page.text, page.title
                    
        return "", ""

    def clean_wiki_text(self, text: str) -> str:
        """Clean and normalize Wikipedia text"""
        # Remove references, links etc
        text = text.replace("==References==", "")
        text = text.replace("==External links==", "")
        
        # Remove duplicate whitespace
        text = " ".join(text.split())
        return text
        
    def rank_evidence_passages(self, claim: str, passages: List[str]) -> List[Tuple[str, float]]:
        """
        Rank evidence passages by relevance to claim using:
        - Semantic similarity (cosine)
        - Keyword overlap
        Returns: List of (passage, score) tuples
        """
        claim_embedding = self.encoder.encode([claim])[0]
        passage_embeddings = self.encoder.encode(passages)
        
        # Compute semantic similarities
        similarities = [np.dot(claim_embedding, p_emb)/(np.linalg.norm(claim_embedding)*np.linalg.norm(p_emb))
                      for p_emb in passage_embeddings]
                      
        # Compute keyword overlaps
        claim_doc = self.nlp(claim)
        claim_keywords = set(t.text.lower() for t in claim_doc if not t.is_stop)
        
        keyword_scores = []
        for p in passages:
            p_doc = self.nlp(p)
            p_keywords = set(t.text.lower() for t in p_doc if not t.is_stop)
            overlap = len(claim_keywords & p_keywords) / len(claim_keywords)
            keyword_scores.append(overlap)
            
        # Combine scores (0.7 semantic + 0.3 keyword)
        final_scores = [0.7*sim + 0.3*key for sim, key in zip(similarities, keyword_scores)]
        
        # Sort by score
        ranked = sorted(zip(passages, final_scores), key=lambda x: x[1], reverse=True)
        return ranked

    def retrieve_evidence(self, claim: str) -> Dict:
        """
        Main evidence retrieval function
        Returns dict with evidence text, source and scores
        """
        # Extract entities for search
        doc = self.nlp(claim)
        entities = [ent.text for ent in doc.ents]
        
        if not entities:
            # Fallback to noun chunks if no entities found
            entities = [chunk.text for chunk in doc.noun_chunks]
            
        # Get evidence text for each entity
        all_evidence = []
        sources = []
        
        for entity in entities:
            text, source = self.get_wiki_text(entity)
            if text:
                text = self.clean_wiki_text(text)
                all_evidence.append(text)
                sources.append(source)
                
        if not all_evidence:
            return {
                "evidence": "",
                "source": "",
                "score": 0.0
            }
            
        # Rank evidence passages
        ranked_evidence = self.rank_evidence_passages(claim, all_evidence)
        
        # Return best evidence
        best_evidence, score = ranked_evidence[0]
        source = sources[all_evidence.index(best_evidence)]
        
        return {
            "evidence": best_evidence,
            "source": f"Wikipedia - {source}",
            "score": float(score)
        }