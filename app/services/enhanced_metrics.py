"""
Enhanced Metrics Calculation Service
=====================================

Provides improved, ML-enhanced calculations for chat metrics:
- Resonant Energy: Semantic coherence + sentiment alignment + structural quality
- Evidence Score: RAG verification + citation quality + factual grounding
- Anchor Following: Semantic similarity + topic modeling + entity tracking
- Context Coherence: Discourse analysis + conversation flow + topic continuity
- Memory Utilization: Actual memory service usage tracking

Uses:
- Sentence embeddings for semantic similarity (via memory_service)
- TF-IDF for keyword extraction
- Sentiment analysis for emotional alignment
- Entity extraction for topic tracking
"""

import re
import math
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)


# ============================================================================
# STOPWORDS - Common words to filter out for better keyword extraction
# ============================================================================
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
    "used", "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "whom", "whose", "where",
    "when", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "also", "now", "here",
    "there", "then", "once", "if", "because", "until", "while", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "any", "your", "my", "his",
    "her", "our", "their", "me", "him", "us", "them", "myself", "yourself",
}


@dataclass
class EnhancedMetricsResult:
    """Complete metrics result with detailed breakdown."""
    resonant_energy: float
    evidence_score: float
    anchor_following: float
    context_coherence: float
    memory_utilization: float
    
    # Detailed breakdowns for transparency
    resonant_energy_breakdown: Dict[str, float] = field(default_factory=dict)
    evidence_breakdown: Dict[str, float] = field(default_factory=dict)
    anchor_breakdown: Dict[str, float] = field(default_factory=dict)
    coherence_breakdown: Dict[str, float] = field(default_factory=dict)
    memory_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Confidence score for the overall calculation
    confidence: float = 0.8


@dataclass
class MemoryUsageRecord:
    """Tracks actual memory service usage for a message."""
    message_id: str
    memories_retrieved: int = 0
    memories_used_in_response: int = 0
    anchor_matches: int = 0
    rag_queries_made: int = 0
    embedding_lookups: int = 0
    total_memory_tokens: int = 0


class EnhancedMetricsCalculator:
    """
    Advanced metrics calculator using NLP and semantic analysis.
    
    Improvements over basic heuristics:
    1. Semantic similarity using cosine distance on word vectors
    2. TF-IDF keyword extraction for topic modeling
    3. Entity tracking across conversation
    4. Actual memory service usage tracking
    5. RAG-based evidence verification
    """
    
    def __init__(self):
        self._memory_usage_tracker: Dict[str, MemoryUsageRecord] = {}
        self._conversation_entities: Dict[str, Set[str]] = {}  # chat_id -> entities

    def _compute_cosine_similarity_embeddings(
        self,
        vec1: Optional[List[float]],
        vec2: Optional[List[float]],
    ) -> float:
        """Compute cosine similarity between two embedding vectors."""
        if not vec1 or not vec2:
            return 0.0
        if len(vec1) != len(vec2):
            return 0.0
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for a, b in zip(vec1, vec2):
            dot += a * b
            norm1 += a * a
            norm2 += b * b
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (math.sqrt(norm1) * math.sqrt(norm2))
    
    # ========================================================================
    # TEXT PROCESSING UTILITIES
    # ========================================================================
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, filtering stopwords."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in STOPWORDS]
    
    def _extract_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Extract n-grams from token list."""
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _compute_tfidf_keywords(
        self, 
        text: str, 
        corpus: List[str], 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Extract top keywords using TF-IDF scoring."""
        tokens = self._tokenize(text)
        if not tokens:
            return []
        
        # Term frequency in document
        tf = Counter(tokens)
        max_tf = max(tf.values()) if tf else 1
        
        # Normalize TF
        tf_normalized = {word: count / max_tf for word, count in tf.items()}
        
        # Document frequency across corpus
        df = Counter()
        for doc in corpus:
            doc_tokens = set(self._tokenize(doc))
            for token in doc_tokens:
                df[token] += 1
        
        # Calculate TF-IDF
        n_docs = len(corpus) + 1  # +1 to avoid division by zero
        tfidf_scores = {}
        for word, tf_score in tf_normalized.items():
            idf = math.log(n_docs / (df.get(word, 0) + 1))
            tfidf_scores[word] = tf_score * idf
        
        # Return top-k keywords
        sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]
    
    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extract named entities and key concepts from text.
        Uses pattern matching for common entity types.
        """
        entities = set()
        
        # Capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update(proper_nouns)
        
        # Technical terms (camelCase, snake_case, ALL_CAPS)
        camel_case = re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', text)
        snake_case = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text)
        all_caps = re.findall(r'\b[A-Z]{2,}\b', text)
        entities.update(camel_case)
        entities.update(snake_case)
        entities.update(all_caps)
        
        # Code-related entities
        code_entities = re.findall(r'`([^`]+)`', text)
        entities.update(code_entities)
        
        # URLs and file paths
        urls = re.findall(r'https?://[^\s]+', text)
        file_paths = re.findall(r'[/\\][\w./\\-]+\.\w+', text)
        entities.update(urls)
        entities.update(file_paths)
        
        return entities
    
    def _compute_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _compute_cosine_similarity_tokens(
        self, 
        tokens1: List[str], 
        tokens2: List[str]
    ) -> float:
        """Compute cosine similarity between two token lists using TF vectors."""
        if not tokens1 or not tokens2:
            return 0.0
        
        # Create TF vectors
        tf1 = Counter(tokens1)
        tf2 = Counter(tokens2)
        
        # Get all unique terms
        all_terms = set(tf1.keys()) | set(tf2.keys())
        
        # Compute dot product and magnitudes
        dot_product = sum(tf1.get(term, 0) * tf2.get(term, 0) for term in all_terms)
        mag1 = math.sqrt(sum(v ** 2 for v in tf1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in tf2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    # ========================================================================
    # MEMORY USAGE TRACKING
    # ========================================================================
    
    def record_memory_usage(
        self,
        message_id: str,
        memories_retrieved: int = 0,
        memories_used: int = 0,
        anchor_matches: int = 0,
        rag_queries: int = 0,
        embedding_lookups: int = 0,
        memory_tokens: int = 0
    ):
        """Record actual memory service usage for a message."""
        self._memory_usage_tracker[message_id] = MemoryUsageRecord(
            message_id=message_id,
            memories_retrieved=memories_retrieved,
            memories_used_in_response=memories_used,
            anchor_matches=anchor_matches,
            rag_queries_made=rag_queries,
            embedding_lookups=embedding_lookups,
            total_memory_tokens=memory_tokens
        )
    
    def get_memory_usage(self, message_id: str) -> Optional[MemoryUsageRecord]:
        """Get memory usage record for a message."""
        return self._memory_usage_tracker.get(message_id)
    
    # ========================================================================
    # ENHANCED METRIC CALCULATIONS
    # ========================================================================
    
    def calculate_resonant_energy(
        self,
        content: str,
        base_resonance_score: float,
        chat_messages: List[Dict[str, Any]],
        message_hash: Optional[str] = None,
        xyz_coords: Optional[Tuple[float, float, float]] = None,
        response_embedding: Optional[List[float]] = None,
        recent_user_embedding: Optional[List[float]] = None,
        rag_sources: Optional[List[Dict[str, Any]]] = None,
        message_id: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Resonant Energy with enhanced NLP analysis.
        
        Components:
        1. Base resonance from Hash Sphere (40%)
        2. Semantic coherence with conversation (25%)
        3. Structural quality assessment (20%)
        4. Sentiment alignment (15%)
        
        Returns: (score, breakdown_dict)
        """
        breakdown = {}
        
        # 1. Base resonance score (40% weight)
        base_score = base_resonance_score if base_resonance_score else 0.5
        breakdown["base_resonance"] = base_score
        
        # 2. Semantic coherence with conversation (25% weight)
        semantic_score = 0.5
        embedding_similarity = self._compute_cosine_similarity_embeddings(
            response_embedding,
            recent_user_embedding,
        )
        if embedding_similarity > 0.0:
            semantic_score = 0.3 + (embedding_similarity * 0.7)
        elif chat_messages:
            # Fallback to token similarity
            user_contents = [
                m.get("content", "")
                for m in chat_messages[-5:]
                if m.get("role") == "user"
            ]
            if user_contents:
                response_tokens = self._tokenize(content)
                user_tokens: List[str] = []
                for uc in user_contents:
                    user_tokens.extend(self._tokenize(uc))
                semantic_score = self._compute_cosine_similarity_tokens(response_tokens, user_tokens)
                semantic_score = 0.3 + (semantic_score * 0.7)
        breakdown["semantic_coherence"] = semantic_score
        
        # 3. Structural quality (20% weight)
        structural_score = self._assess_structural_quality(content)
        breakdown["structural_quality"] = structural_score
        
        # 4. Sentiment alignment (15% weight)
        sentiment_score = self._assess_sentiment_alignment(content, chat_messages)
        breakdown["sentiment_alignment"] = sentiment_score
        
        # Bonus for Hash Sphere integration
        integration_bonus = 0.0
        if message_hash:
            integration_bonus += 0.03
        if xyz_coords and xyz_coords[0] is not None:
            integration_bonus += 0.02
        # Boost when actual Hash Sphere memories were used
        if rag_sources:
            integration_bonus += min(0.35, len(rag_sources) * 0.05)
        # Check memory usage tracker for this message
        mem_record = self.get_memory_usage(message_id)
        if mem_record and mem_record.memories_retrieved > 0:
            integration_bonus += min(0.20, mem_record.memories_retrieved * 0.02)
        breakdown["hash_sphere_integration"] = min(1.0, integration_bonus)
        
        # Weighted combination
        final_score = (
            base_score * 0.40 +
            semantic_score * 0.25 +
            structural_score * 0.20 +
            sentiment_score * 0.15 +
            integration_bonus
        )
        
        return max(0.1, min(1.0, final_score)), breakdown
    
    def _assess_structural_quality(self, content: str) -> float:
        """Assess the structural quality of a response."""
        score = 0.5
        
        # Code blocks (strong indicator of technical depth)
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        if code_blocks > 0:
            score += 0.15 * min(code_blocks, 3)
        
        # Inline code
        inline_code = len(re.findall(r'`[^`]+`', content))
        if inline_code > 0:
            score += 0.05 * min(inline_code / 5, 1.0)
        
        # Lists (organized thinking)
        list_items = len(re.findall(r'^[\s]*[-*•]\s', content, re.MULTILINE))
        numbered_items = len(re.findall(r'^[\s]*\d+[.)]\s', content, re.MULTILINE))
        if list_items + numbered_items > 0:
            score += 0.08 * min((list_items + numbered_items) / 5, 1.0)
        
        # Headers (structured response)
        headers = len(re.findall(r'^#{1,6}\s', content, re.MULTILINE))
        if headers > 0:
            score += 0.05 * min(headers / 3, 1.0)
        
        # Appropriate length
        length = len(content)
        if 200 <= length <= 3000:
            score += 0.05
        elif length < 50:
            score -= 0.1
        elif length > 5000:
            score -= 0.05  # Might be too verbose
        
        # Paragraph structure
        paragraphs = len(re.findall(r'\n\n', content))
        if 1 <= paragraphs <= 10:
            score += 0.03
        
        return max(0.1, min(1.0, score))
    
    def _assess_sentiment_alignment(
        self, 
        content: str, 
        chat_messages: List[Dict[str, Any]]
    ) -> float:
        """
        Assess how well the response sentiment aligns with conversation context.
        """
        # Simple sentiment indicators
        positive_words = {
            "great", "excellent", "good", "helpful", "perfect", "thanks",
            "wonderful", "amazing", "awesome", "fantastic", "love", "best"
        }
        negative_words = {
            "bad", "wrong", "error", "fail", "problem", "issue", "broken",
            "terrible", "awful", "hate", "worst", "difficult", "frustrated"
        }
        professional_words = {
            "however", "therefore", "additionally", "furthermore", "specifically",
            "regarding", "concerning", "accordingly", "consequently"
        }
        
        content_lower = content.lower()
        content_words = set(content_lower.split())
        
        # Count sentiment indicators
        positive_count = len(content_words & positive_words)
        negative_count = len(content_words & negative_words)
        professional_count = len(content_words & professional_words)
        
        # Check user sentiment in recent messages
        user_frustrated = False
        user_positive = False
        for msg in chat_messages[-3:]:
            if msg.get("role") == "user":
                msg_lower = msg.get("content", "").lower()
                if any(w in msg_lower for w in ["frustrated", "angry", "annoyed", "help", "please"]):
                    user_frustrated = True
                if any(w in msg_lower for w in ["thanks", "great", "perfect", "awesome"]):
                    user_positive = True
        
        score = 0.5
        
        # Professional tone is generally good
        if professional_count > 0:
            score += 0.1 * min(professional_count / 3, 1.0)
        
        # If user is frustrated, empathetic response is better
        if user_frustrated:
            empathy_markers = ["understand", "sorry", "help", "let me", "here's"]
            if any(m in content_lower for m in empathy_markers):
                score += 0.15
        
        # Match positive energy
        if user_positive and positive_count > 0:
            score += 0.1
        
        # Avoid excessive negativity unless addressing errors
        if negative_count > 3 and "error" not in content_lower and "bug" not in content_lower:
            score -= 0.1
        
        return max(0.1, min(1.0, score))
    
    def calculate_evidence_score(
        self,
        content: str,
        message_hash: Optional[str],
        anchors: List[str],
        rag_sources: Optional[List[Dict[str, Any]]] = None,
        memory_matches: int = 0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Evidence Score with RAG verification.
        
        Components:
        1. Citation quality (25%)
        2. Specificity of claims (25%)
        3. RAG source grounding (25%)
        4. Memory anchor utilization (25%)
        
        Returns: (score, breakdown_dict)
        """
        breakdown = {}
        
        if not content:
            return 0.5, {"empty_content": True}
        
        content_lower = content.lower()
        
        # 1. Citation quality (25%)
        citation_score = self._assess_citation_quality(content)
        breakdown["citation_quality"] = citation_score
        
        # 2. Specificity of claims (25%)
        specificity_score = self._assess_specificity(content)
        breakdown["specificity"] = specificity_score
        
        # 3. RAG source grounding (25%)
        rag_score = 0.3  # Base score
        if rag_sources:
            # More sources = better grounding
            source_count = len(rag_sources)
            rag_score = 0.3 + (0.7 * min(source_count / 5, 1.0))
            
            # Check if sources are actually referenced
            source_texts = [s.get("content", "")[:100].lower() for s in rag_sources]
            referenced = sum(1 for st in source_texts if any(w in content_lower for w in st.split()[:5]))
            if referenced > 0:
                rag_score += 0.1 * min(referenced / len(rag_sources), 1.0)
        breakdown["rag_grounding"] = min(1.0, rag_score)
        
        # 4. Memory anchor utilization (25%)
        anchor_score = 0.3
        if anchors:
            anchor_count = len(anchors)
            anchor_score = 0.3 + (0.5 * min(anchor_count / 10, 1.0))
            
            # Check if anchors are reflected in response
            anchor_words = set()
            for anchor in anchors:
                anchor_words.update(self._tokenize(anchor))
            response_words = set(self._tokenize(content))
            overlap = len(anchor_words & response_words)
            if overlap > 0:
                anchor_score += 0.2 * min(overlap / max(len(anchor_words), 1), 1.0)
        
        # Bonus for hash presence
        if message_hash:
            anchor_score += 0.05
        breakdown["anchor_utilization"] = min(1.0, anchor_score)
        
        # Weighted combination
        final_score = (
            citation_score * 0.25 +
            specificity_score * 0.25 +
            breakdown["rag_grounding"] * 0.25 +
            breakdown["anchor_utilization"] * 0.25
        )
        
        return max(0.1, min(1.0, final_score)), breakdown
    
    def _assess_citation_quality(self, content: str) -> float:
        """Assess the quality of citations and references."""
        score = 0.3
        content_lower = content.lower()
        
        # Strong citation markers
        strong_citations = [
            "according to the documentation",
            "as stated in",
            "the official",
            "from the source",
            "based on the code",
            "as shown in",
        ]
        strong_count = sum(1 for c in strong_citations if c in content_lower)
        score += 0.15 * min(strong_count, 3)
        
        # Moderate citation markers
        moderate_citations = [
            "according to", "based on", "as mentioned",
            "for example", "specifically", "in this case",
        ]
        moderate_count = sum(1 for c in moderate_citations if c in content_lower)
        score += 0.08 * min(moderate_count, 4)
        
        # Code references
        if "```" in content:
            score += 0.1
        
        # URLs
        if re.search(r'https?://', content):
            score += 0.1
        
        # File paths
        if re.search(r'[/\\][\w./\\-]+\.\w+', content):
            score += 0.05
        
        return min(1.0, score)
    
    def _assess_specificity(self, content: str) -> float:
        """Assess how specific vs vague the claims are."""
        score = 0.5
        content_lower = content.lower()
        
        # Specific indicators (increase score)
        specific_patterns = [
            r'\b\d+\.\d+\b',  # Version numbers
            r'\b\d+\s*(ms|seconds?|minutes?|hours?|bytes?|kb|mb|gb)\b',  # Measurements
            r'\b(line|column|row)\s*\d+\b',  # Line numbers
            r'\b(step|option|method)\s*\d+\b',  # Numbered items
            r'`[^`]+`',  # Inline code
        ]
        for pattern in specific_patterns:
            matches = len(re.findall(pattern, content_lower))
            if matches > 0:
                score += 0.05 * min(matches, 3)
        
        # Vague indicators (decrease score)
        vague_words = [
            "maybe", "perhaps", "probably", "might", "could be",
            "i think", "i believe", "possibly", "sometimes", "often",
            "generally", "usually", "typically", "sort of", "kind of",
        ]
        vague_count = sum(1 for v in vague_words if v in content_lower)
        score -= 0.05 * min(vague_count, 4)
        
        # Hedging language
        hedging = ["it depends", "not sure", "hard to say", "difficult to"]
        if any(h in content_lower for h in hedging):
            score -= 0.1
        
        return max(0.1, min(1.0, score))
    
    def calculate_anchor_following(
        self,
        content: str,
        chat_messages: List[Dict[str, Any]],
        message_role: str,
        stored_anchors: List[str],
        resonance_score: Optional[float] = None,
        response_embedding: Optional[List[float]] = None,
        recent_user_embedding: Optional[List[float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Anchor Following with semantic similarity.
        
        Components:
        1. Semantic similarity to user queries (30%)
        2. Entity continuity tracking (25%)
        3. Topic keyword overlap (25%)
        4. Direct reference markers (20%)
        
        Returns: (score, breakdown_dict)
        """
        breakdown = {}
        
        if message_role != "assistant" or not chat_messages:
            return 0.5, {"not_applicable": True}
        
        content_lower = content.lower()
        response_tokens = self._tokenize(content)
        
        # 1. Semantic similarity to recent user queries (30%)
        user_messages = [
            m.get("content", "") 
            for m in chat_messages[-5:] 
            if m.get("role") == "user"
        ]
        
        semantic_score = 0.4
        embedding_similarity = self._compute_cosine_similarity_embeddings(
            response_embedding,
            recent_user_embedding,
        )
        if embedding_similarity > 0.0:
            semantic_score = 0.3 + (embedding_similarity * 0.7)
        elif user_messages:
            similarities = []
            for user_msg in user_messages:
                user_tokens = self._tokenize(user_msg)
                sim = self._compute_cosine_similarity_tokens(response_tokens, user_tokens)
                similarities.append(sim)
            if similarities:
                semantic_score = 0.3 + (max(similarities) * 0.7)
        breakdown["semantic_similarity"] = semantic_score
        
        # 2. Entity continuity (25%)
        # Extract entities from user messages
        user_entities = set()
        for msg in user_messages:
            user_entities.update(self._extract_entities(msg))
        
        response_entities = self._extract_entities(content)
        entity_overlap = self._compute_jaccard_similarity(user_entities, response_entities)
        entity_score = 0.3 + (entity_overlap * 0.7)
        breakdown["entity_continuity"] = entity_score
        
        # 3. Topic keyword overlap (25%)
        user_keywords = set()
        for msg in user_messages:
            tokens = self._tokenize(msg)
            user_keywords.update(tokens[:20])  # Top 20 keywords
        
        response_keywords = set(response_tokens[:30])
        keyword_overlap = len(user_keywords & response_keywords)
        keyword_score = 0.3 + (0.7 * min(keyword_overlap / max(len(user_keywords), 1), 1.0))
        breakdown["keyword_overlap"] = keyword_score
        
        # 4. Direct reference markers (20%)
        reference_markers = [
            "you asked", "your question", "you mentioned", "as you said",
            "regarding your", "to answer your", "you're asking about",
            "based on what you", "from your", "in your message",
        ]
        ref_count = sum(1 for m in reference_markers if m in content_lower)
        reference_score = 0.3 + (0.7 * min(ref_count / 3, 1.0))
        breakdown["direct_references"] = reference_score
        
        # Bonus for stored anchor utilization
        anchor_bonus = 0.0
        if stored_anchors:
            anchor_words = set()
            for anchor in stored_anchors:
                anchor_words.update(self._tokenize(anchor))
            anchor_matches = len(anchor_words & set(response_tokens))
            anchor_bonus = 0.05 * min(anchor_matches / max(len(anchor_words), 1), 1.0)
        breakdown["anchor_bonus"] = anchor_bonus
        
        # Weighted combination
        final_score = (
            semantic_score * 0.30 +
            entity_score * 0.25 +
            keyword_score * 0.25 +
            reference_score * 0.20 +
            anchor_bonus
        )
        
        return max(0.1, min(1.0, final_score)), breakdown
    
    def calculate_context_coherence(
        self,
        content: str,
        chat_messages: List[Dict[str, Any]],
        message_role: str,
        response_embedding: Optional[List[float]] = None,
        all_user_embedding: Optional[List[float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Context Coherence with discourse analysis.
        
        Components:
        1. Topic continuity across conversation (30%)
        2. Logical flow indicators (25%)
        3. Question-answer alignment (25%)
        4. Conversation progression (20%)
        
        Returns: (score, breakdown_dict)
        """
        breakdown = {}
        
        if message_role != "assistant" or len(chat_messages) < 2:
            return 0.5, {"not_applicable": True}
        
        content_lower = content.lower()
        response_tokens = self._tokenize(content)
        
        # 1. Topic continuity (30%)
        embedding_similarity = self._compute_cosine_similarity_embeddings(
            response_embedding,
            all_user_embedding,
        )

        if embedding_similarity > 0.0:
            topic_score = 0.3 + (embedding_similarity * 0.7)
            breakdown["topic_continuity"] = topic_score
        else:
            # Fallback to token-based topic continuity
            all_user_tokens = []
            all_assistant_tokens = []
            for msg in chat_messages:
                tokens = self._tokenize(msg.get("content", ""))
                if msg.get("role") == "user":
                    all_user_tokens.extend(tokens)
                else:
                    all_assistant_tokens.extend(tokens)
            
            # Calculate topic overlap
            conversation_topics = set(all_user_tokens[:50])  # Top topics from users
            response_topics = set(response_tokens[:30])
            topic_overlap = self._compute_jaccard_similarity(conversation_topics, response_topics)
            topic_score = 0.3 + (topic_overlap * 0.7)
            breakdown["topic_continuity"] = topic_score
        
        # 2. Logical flow indicators (25%)
        flow_markers = {
            "continuation": ["continuing", "furthermore", "additionally", "also", "moreover"],
            "contrast": ["however", "but", "although", "nevertheless", "on the other hand"],
            "causation": ["therefore", "thus", "consequently", "as a result", "because"],
            "conclusion": ["in summary", "to conclude", "finally", "in conclusion"],
            "elaboration": ["specifically", "in particular", "for example", "namely"],
        }
        
        flow_score = 0.4
        for category, markers in flow_markers.items():
            if any(m in content_lower for m in markers):
                flow_score += 0.1
        breakdown["logical_flow"] = min(1.0, flow_score)
        
        # 3. Question-answer alignment (25%)
        qa_score = 0.5
        recent_questions = [
            m.get("content", "") 
            for m in chat_messages[-3:] 
            if m.get("role") == "user" and m.get("content", "").strip().endswith("?")
        ]
        
        if recent_questions:
            # Check if response addresses the question
            answer_indicators = [
                "here", "this", "the answer", "you can", "to do this",
                "yes", "no", "it is", "it's", "that's", "the reason",
            ]
            if any(ind in content_lower for ind in answer_indicators):
                qa_score += 0.2
            
            # Check for question keywords in response
            for q in recent_questions:
                q_tokens = set(self._tokenize(q))
                r_tokens = set(response_tokens)
                if len(q_tokens & r_tokens) > 2:
                    qa_score += 0.15
                    break
        breakdown["qa_alignment"] = min(1.0, qa_score)
        
        # 4. Conversation progression (20%)
        progression_score = 0.5
        
        # Check for building on previous responses
        if len(chat_messages) >= 2:
            prev_assistant = None
            for msg in reversed(chat_messages[:-1]):
                if msg.get("role") == "assistant":
                    prev_assistant = msg.get("content", "")
                    break
            
            if prev_assistant:
                prev_tokens = set(self._tokenize(prev_assistant)[:20])
                current_tokens = set(response_tokens[:20])
                continuity = len(prev_tokens & current_tokens)
                if continuity > 3:
                    progression_score += 0.2
        
        # Penalize off-topic indicators
        off_topic = ["by the way", "unrelated", "different topic", "changing subject", "off topic"]
        if any(ot in content_lower for ot in off_topic):
            progression_score -= 0.2
        breakdown["progression"] = max(0.1, min(1.0, progression_score))
        
        # Weighted combination
        final_score = (
            topic_score * 0.30 +
            breakdown["logical_flow"] * 0.25 +
            breakdown["qa_alignment"] * 0.25 +
            breakdown["progression"] * 0.20
        )
        
        return max(0.1, min(1.0, final_score)), breakdown
    
    def calculate_memory_utilization(
        self,
        message_id: str,
        anchors: List[str],
        rag_sources: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Memory Utilization based on actual memory service usage.
        
        Components:
        1. Actual memory service calls (40%)
        2. RAG sources used (30%)
        3. Anchor count and quality (30%)
        
        Returns: (score, breakdown_dict)
        """
        breakdown = {}
        
        # 1. Actual memory service usage (40%)
        memory_record = self.get_memory_usage(message_id)
        memory_service_score = 0.0
        
        if memory_record:
            # Memories retrieved
            if memory_record.memories_retrieved > 0:
                memory_service_score += 0.3 * min(memory_record.memories_retrieved / 10, 1.0)
            
            # Memories actually used
            if memory_record.memories_used_in_response > 0:
                memory_service_score += 0.4 * min(memory_record.memories_used_in_response / 5, 1.0)
            
            # RAG queries made
            if memory_record.rag_queries_made > 0:
                memory_service_score += 0.2 * min(memory_record.rag_queries_made / 3, 1.0)
            
            # Embedding lookups
            if memory_record.embedding_lookups > 0:
                memory_service_score += 0.1
            
            breakdown["actual_memory_calls"] = memory_service_score
        else:
            breakdown["actual_memory_calls"] = 0.0
        
        # 2. RAG sources (30%)
        rag_score = 0.0
        if rag_sources:
            source_count = len(rag_sources)
            rag_score = min(source_count / 5, 1.0)
            
            # Quality bonus for high-scoring sources
            high_quality = sum(1 for s in rag_sources if s.get("score", 0) > 0.7)
            if high_quality > 0:
                rag_score += 0.1 * min(high_quality / 3, 1.0)
        breakdown["rag_sources"] = min(1.0, rag_score)
        
        # 3. Anchor utilization (30%)
        anchor_score = 0.0
        if anchors:
            anchor_count = len(anchors)
            anchor_score = min(anchor_count / 10, 1.0)
        breakdown["anchor_count"] = anchor_score
        
        # Weighted combination
        final_score = (
            breakdown["actual_memory_calls"] * 0.40 +
            breakdown["rag_sources"] * 0.30 +
            breakdown["anchor_count"] * 0.30
        )
        
        return max(0.0, min(1.0, final_score)), breakdown
    
    def calculate_all_metrics(
        self,
        content: str,
        message_id: str,
        message_role: str,
        base_resonance_score: float,
        chat_messages: List[Dict[str, Any]],
        message_hash: Optional[str] = None,
        xyz_coords: Optional[Tuple[float, float, float]] = None,
        anchors: Optional[List[str]] = None,
        rag_sources: Optional[List[Dict[str, Any]]] = None,
        response_embedding: Optional[List[float]] = None,
        recent_user_embedding: Optional[List[float]] = None,
        all_user_embedding: Optional[List[float]] = None,
    ) -> EnhancedMetricsResult:
        """
        Calculate all enhanced metrics for a message.
        
        Returns comprehensive EnhancedMetricsResult with all scores and breakdowns.
        """
        anchors = anchors or []
        
        # Calculate each metric
        resonant_energy, re_breakdown = self.calculate_resonant_energy(
            content,
            base_resonance_score,
            chat_messages,
            message_hash,
            xyz_coords,
            response_embedding=response_embedding,
            recent_user_embedding=recent_user_embedding,
            rag_sources=rag_sources,
            message_id=message_id,
        )
        
        evidence_score, ev_breakdown = self.calculate_evidence_score(
            content, message_hash, anchors, rag_sources
        )
        
        anchor_following, af_breakdown = self.calculate_anchor_following(
            content,
            chat_messages,
            message_role,
            anchors,
            base_resonance_score,
            response_embedding=response_embedding,
            recent_user_embedding=recent_user_embedding,
        )
        
        context_coherence, cc_breakdown = self.calculate_context_coherence(
            content,
            chat_messages,
            message_role,
            response_embedding=response_embedding,
            all_user_embedding=all_user_embedding,
        )
        
        memory_utilization, mu_breakdown = self.calculate_memory_utilization(
            message_id, anchors, rag_sources
        )
        
        # Calculate overall confidence based on data availability
        confidence = 0.8
        if not chat_messages:
            confidence -= 0.2
        if not anchors:
            confidence -= 0.1
        if not rag_sources:
            confidence -= 0.1
        
        return EnhancedMetricsResult(
            resonant_energy=resonant_energy,
            evidence_score=evidence_score,
            anchor_following=anchor_following,
            context_coherence=context_coherence,
            memory_utilization=memory_utilization,
            resonant_energy_breakdown=re_breakdown,
            evidence_breakdown=ev_breakdown,
            anchor_breakdown=af_breakdown,
            coherence_breakdown=cc_breakdown,
            memory_breakdown=mu_breakdown,
            confidence=max(0.3, confidence)
        )


# Global instance
enhanced_metrics_calculator = EnhancedMetricsCalculator()
