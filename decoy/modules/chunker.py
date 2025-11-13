from typing import List, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chunker:
    def __init__(self):
        """Initialize chunker with models and parameters"""
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def get_chunk_size(self, text: str, base_size: int = 3) -> int:
        """
        Dynamically determine chunk size based on:
        - Average sentence length
        - Presence of numbers/dates
        - Text complexity
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return base_size
            
        # Get average sentence length
        avg_len = np.mean([len(s.split()) for s in sentences])
        
        # Adjust base size based on average length
        if avg_len < 10:
            chunk_size = base_size + 1
        elif avg_len > 25:
            chunk_size = base_size - 1
        else:
            chunk_size = base_size
            
        # Ensure chunk size is at least 2
        return max(2, chunk_size)

    def make_overlapping_chunks(self, text: str, claim: str) -> List[str]:
        """
        Create overlapping chunks with dynamic sizing
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return []
            
        # Get dynamic chunk size
        chunk_size = self.get_chunk_size(text)
        
        # Create overlapping chunks
        chunks = []
        for i in range(0, len(sentences)):
            # Get chunk_size sentences starting at i
            chunk = sentences[i:i + chunk_size]
            if chunk:
                chunks.append(" ".join(chunk))
                
        # Add sliding chunks with 50% overlap
        slide = chunk_size // 2
        for i in range(slide, len(sentences), slide):
            chunk = sentences[i:i + chunk_size]
            if chunk and " ".join(chunk) not in chunks:
                chunks.append(" ".join(chunk))
                
        return chunks

    def compute_semantic_similarity(self, claim: str, chunk: str) -> float:
        """Compute semantic similarity between claim and chunk"""
        claim_emb = self.encoder.encode([claim])[0]
        chunk_emb = self.encoder.encode([chunk])[0]
        
        similarity = np.dot(claim_emb, chunk_emb) / (np.linalg.norm(claim_emb) * np.linalg.norm(chunk_emb))
        return float(similarity)

    def retain_relevant_chunks(self, chunks: List[str], claim: str, 
                             min_similarity: float = 0.4) -> List[Dict]:
        """
        Filter chunks based on semantic similarity with claim
        Returns list of dicts with chunks and their similarity scores
        """
        chunk_scores = []
        
        for chunk in chunks:
            similarity = self.compute_semantic_similarity(claim, chunk)
            if similarity >= min_similarity:
                chunk_scores.append({
                    "chunk": chunk,
                    "similarity": similarity
                })
                
        # Sort by similarity
        chunk_scores.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Keep top 5 most relevant chunks
        return chunk_scores[:5]

    def process_text(self, text: str, claim: str) -> List[Dict]:
        """
        Main chunking function that:
        1. Creates overlapping chunks
        2. Retains relevant chunks
        3. Returns chunks with similarity scores
        """
        logger.info(f"Processing text for claim: {claim}")
        
        # Create chunks
        chunks = self.make_overlapping_chunks(text, claim)
        logger.info(f"Created {len(chunks)} initial chunks")
        
        # Filter and score chunks
        relevant_chunks = self.retain_relevant_chunks(chunks, claim)
        logger.info(f"Retained {len(relevant_chunks)} relevant chunks")
        
        return relevant_chunks