from typing import Dict, List, Tuple, Optional
import spacy
import numpy as np
from fuzzywuzzy import fuzz
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Scorer:
    def __init__(self):
        """Initialize scorer with required models"""
        self.nlp = spacy.load('en_core_web_lg')
        # Load NLI model for entailment scoring
        self.nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
        self.nli_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
        
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity using spaCy doc vectors
        """
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        if doc1.vector_norm and doc2.vector_norm:
            return float(doc1.similarity(doc2))
        return 0.0

    def compute_keyword_overlap(self, text1: str, text2: str) -> float:
        """
        Compute keyword overlap with higher weights for named entities
        """
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        # Get keywords (non-stop words)
        keywords1 = set(t.text.lower() for t in doc1 if not t.is_stop)
        keywords2 = set(t.text.lower() for t in doc2 if not t.is_stop)
        
        # Get named entities
        ents1 = set(ent.text.lower() for ent in doc1.ents)
        ents2 = set(ent.text.lower() for ent in doc2.ents)
        
        # Compute overlaps
        keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1), 1)
        entity_overlap = len(ents1 & ents2) / max(len(ents1), 1) if ents1 else 0
        
        # Weight entity overlap higher (0.7) than general keyword overlap (0.3)
        return 0.3 * keyword_overlap + 0.7 * entity_overlap

    def extract_numbers(self, text: str) -> List[str]:
        """Extract numbers and dates from text"""
        # Pattern for numbers, years, dates
        number_pattern = r'\b\d+(?:[\.,]\d+)?(?:st|nd|rd|th)?\b|\b(?:19|20)\d{2}\b'
        return re.findall(number_pattern, text)

    def compute_numeric_match(self, text1: str, text2: str) -> float:
        """
        Compute match score for numbers and dates
        """
        nums1 = self.extract_numbers(text1)
        nums2 = self.extract_numbers(text2)
        
        if not nums1 or not nums2:
            return 0.0
            
        # Compare each number pair
        matches = 0
        total = len(nums1)
        
        for n1 in nums1:
            for n2 in nums2:
                # Exact match
                if n1 == n2:
                    matches += 1
                    break
                # Close numeric match (within 5% for non-years)
                try:
                    num1 = float(n1)
                    num2 = float(n2)
                    if abs(num1 - num2) / max(num1, 1) < 0.05:
                        matches += 0.8  # Partial credit for close match
                        break
                except ValueError:
                    continue
                    
        return matches / total if total > 0 else 0.0

    def compute_nli_score(self, claim: str, evidence: str) -> float:
        """
        Compute natural language inference score
        """
        # Prepare input
        inputs = self.nli_tokenizer(claim, evidence, return_tensors="pt", 
                                  truncation=True, max_length=512)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
        # Get entailment probability (assuming order: contradiction, neutral, entailment)
        entail_score = float(probs[0][2])  # Probability of entailment class
        return entail_score

    def get_explanation(self, claim: str, evidence: str, scores: Dict[str, float],
                       verdict: str) -> str:
        """
        Generate detailed explanation for the verdict
        """
        explanation_parts = []
        
        # Add evidence context
        evidence_preview = evidence[:200] + "..." if len(evidence) > 200 else evidence
        explanation_parts.append(f"Based on evidence: '{evidence_preview}'")
        
        # Add score explanations
        highest_score = max(scores.items(), key=lambda x: x[1])
        explanation_parts.append(f"Strongest signal came from {highest_score[0]} "
                               f"(score: {highest_score[1]:.2f})")
        
        # Add specific features that contributed
        if scores['numeric_match'] > 0.8:
            nums_claim = self.extract_numbers(claim)
            nums_evidence = self.extract_numbers(evidence)
            explanation_parts.append(f"Exact number matches found: {set(nums_claim) & set(nums_evidence)}")
            
        doc_claim = self.nlp(claim)
        doc_evidence = self.nlp(evidence)
        matching_ents = set(e.text for e in doc_claim.ents) & set(e.text for e in doc_evidence.ents)
        if matching_ents:
            explanation_parts.append(f"Matching entities: {matching_ents}")
            
        # Add verdict-specific reasoning
        if verdict == "SUPPORTED":
            if scores['nli_score'] > 0.8:
                explanation_parts.append("Strong logical entailment detected")
            if scores['semantic_similarity'] > 0.8:
                explanation_parts.append("High semantic similarity with evidence")
                
        elif verdict == "REFUTED":
            # Check for contradicting numbers
            nums_claim = self.extract_numbers(claim)
            nums_evidence = self.extract_numbers(evidence)
            if nums_claim and nums_evidence and not (set(nums_claim) & set(nums_evidence)):
                explanation_parts.append("Contradicting numbers found")
                
            # Check for contradicting entities
            if matching_ents:
                explanation_parts.append("Contradicting context for matching entities")
                
        return " | ".join(explanation_parts)

    def score_evidence(self, claim: str, evidence: str) -> Dict:
        """
        Main scoring function that combines all signals
        Returns dict with scores and explanation
        """
        # Compute individual scores
        scores = {
            'semantic_similarity': self.compute_semantic_similarity(claim, evidence),
            'keyword_overlap': self.compute_keyword_overlap(claim, evidence),
            'numeric_match': self.compute_numeric_match(claim, evidence),
            'nli_score': self.compute_nli_score(claim, evidence)
        }
        
        # Weighted combination
        weights = {
            'semantic_similarity': 0.4,
            'keyword_overlap': 0.3,
            'numeric_match': 0.1,
            'nli_score': 0.2
        }
        
        final_score = sum(score * weights[key] for key, score in scores.items())
        
        # Determine verdict
        if final_score >= 0.7:
            verdict = "SUPPORTED"
        elif final_score <= 0.3:
            verdict = "REFUTED"
        else:
            verdict = "NOT_ENOUGH_EVIDENCE"
            
        # Generate explanation
        explanation = self.get_explanation(claim, evidence, scores, verdict)
        
        return {
            'scores': scores,
            'final_score': final_score,
            'verdict': verdict,
            'explanation': explanation
        }