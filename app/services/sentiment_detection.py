"""
Sentiment and Emotion Detection Service

Analyzes user messages to detect emotional state and adjust response tone accordingly.
"""

import re
from typing import Dict, Optional, Tuple
from enum import Enum


class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    EXCITED = "excited"
    ANXIOUS = "anxious"
    CURIOUS = "curious"
    GRATEFUL = "grateful"
    DISAPPOINTED = "disappointed"


class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# Emotion keywords and patterns
EMOTION_PATTERNS = {
    Emotion.FRUSTRATED: [
        r"\b(frustrated|annoying|annoyed|angry|mad|hate|ugh|argh|damn|wtf|broken|doesn't work|not working|still not|why won't|can't believe)\b",
        r"!{2,}",  # Multiple exclamation marks
        r"\?{2,}",  # Multiple question marks
    ],
    Emotion.CONFUSED: [
        r"\b(confused|don't understand|what do you mean|unclear|lost|help me understand|i'm not sure|what is|how does|explain)\b",
        r"\?$",  # Ends with question mark
    ],
    Emotion.EXCITED: [
        r"\b(excited|awesome|amazing|great|fantastic|love it|perfect|excellent|wonderful|brilliant)\b",
        r"!{1,}.*!{1,}",  # Multiple sentences with exclamation
    ],
    Emotion.ANXIOUS: [
        r"\b(worried|anxious|nervous|urgent|asap|deadline|hurry|quickly|emergency|critical)\b",
    ],
    Emotion.CURIOUS: [
        r"\b(curious|wondering|interested|tell me more|how about|what if|could you explain|i want to know)\b",
    ],
    Emotion.GRATEFUL: [
        r"\b(thanks|thank you|appreciate|grateful|helpful|you're the best|saved me)\b",
    ],
    Emotion.HAPPY: [
        r"\b(happy|glad|pleased|nice|good|yay|hooray|finally)\b",
        r":\)|😊|😄|🎉",
    ],
    Emotion.DISAPPOINTED: [
        r"\b(disappointed|sad|unfortunately|too bad|wish|hoped|expected)\b",
        r":\(|😞|😔",
    ],
}

# Sentiment keywords
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "fantastic", "wonderful",
    "perfect", "love", "like", "best", "helpful", "thanks", "thank", "appreciate",
    "brilliant", "outstanding", "superb", "nice", "happy", "glad", "pleased",
    "excited", "beautiful", "impressive", "remarkable", "incredible"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "dislike", "worst", "broken",
    "error", "bug", "issue", "problem", "fail", "failed", "wrong", "incorrect",
    "frustrated", "annoying", "annoyed", "disappointed", "sad", "angry", "mad",
    "confused", "unclear", "difficult", "hard", "impossible", "useless"
}


def detect_emotion(text: str) -> Tuple[Emotion, float]:
    """
    Detect the primary emotion in the text.
    
    Returns:
        Tuple of (Emotion, confidence_score)
    """
    text_lower = text.lower()
    emotion_scores: Dict[Emotion, float] = {e: 0.0 for e in Emotion}
    
    for emotion, patterns in EMOTION_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            emotion_scores[emotion] += len(matches) * 0.3
    
    # Find the emotion with highest score
    max_emotion = Emotion.NEUTRAL
    max_score = 0.0
    
    for emotion, score in emotion_scores.items():
        if score > max_score:
            max_score = score
            max_emotion = emotion
    
    # Normalize confidence (0-1)
    confidence = min(max_score, 1.0)
    
    return max_emotion, confidence


def detect_sentiment(text: str) -> Tuple[Sentiment, float]:
    """
    Detect overall sentiment of the text.
    
    Returns:
        Tuple of (Sentiment, confidence_score)
    """
    words = set(re.findall(r'\b\w+\b', text.lower()))
    
    positive_count = len(words & POSITIVE_WORDS)
    negative_count = len(words & NEGATIVE_WORDS)
    total = positive_count + negative_count
    
    if total == 0:
        return Sentiment.NEUTRAL, 0.5
    
    if positive_count > negative_count:
        confidence = positive_count / total
        return Sentiment.POSITIVE, confidence
    elif negative_count > positive_count:
        confidence = negative_count / total
        return Sentiment.NEGATIVE, confidence
    else:
        return Sentiment.NEUTRAL, 0.5


def get_response_tone(emotion: Emotion, sentiment: Sentiment) -> Dict[str, any]:
    """
    Get recommended response tone based on detected emotion and sentiment.
    
    Returns:
        Dict with tone recommendations for the AI response
    """
    tone_config = {
        "empathy_level": "normal",  # low, normal, high
        "detail_level": "normal",   # brief, normal, detailed
        "formality": "normal",      # casual, normal, formal
        "encouragement": False,
        "patience": "normal",       # low, normal, high
        "suggested_prefix": None,
    }
    
    # Adjust based on emotion
    if emotion == Emotion.FRUSTRATED:
        tone_config["empathy_level"] = "high"
        tone_config["patience"] = "high"
        tone_config["suggested_prefix"] = "I understand this can be frustrating. Let me help you with that."
        tone_config["detail_level"] = "detailed"
    
    elif emotion == Emotion.CONFUSED:
        tone_config["detail_level"] = "detailed"
        tone_config["patience"] = "high"
        tone_config["suggested_prefix"] = "Let me explain this step by step."
    
    elif emotion == Emotion.ANXIOUS:
        tone_config["empathy_level"] = "high"
        tone_config["suggested_prefix"] = "I'll help you get this done quickly."
        tone_config["detail_level"] = "brief"
    
    elif emotion == Emotion.EXCITED:
        tone_config["encouragement"] = True
        tone_config["formality"] = "casual"
    
    elif emotion == Emotion.CURIOUS:
        tone_config["detail_level"] = "detailed"
        tone_config["encouragement"] = True
    
    elif emotion == Emotion.GRATEFUL:
        tone_config["formality"] = "casual"
        tone_config["suggested_prefix"] = "You're welcome!"
    
    elif emotion == Emotion.DISAPPOINTED:
        tone_config["empathy_level"] = "high"
        tone_config["encouragement"] = True
        tone_config["suggested_prefix"] = "I'm sorry to hear that. Let's see how we can improve this."
    
    return tone_config


def analyze_message(text: str) -> Dict[str, any]:
    """
    Full sentiment and emotion analysis of a message.
    
    Returns:
        Dict with emotion, sentiment, confidence scores, and tone recommendations
    """
    emotion, emotion_confidence = detect_emotion(text)
    sentiment, sentiment_confidence = detect_sentiment(text)
    tone = get_response_tone(emotion, sentiment)
    
    return {
        "emotion": emotion.value,
        "emotion_confidence": emotion_confidence,
        "sentiment": sentiment.value,
        "sentiment_confidence": sentiment_confidence,
        "tone_recommendations": tone,
        "should_prioritize_empathy": emotion in [Emotion.FRUSTRATED, Emotion.ANXIOUS, Emotion.DISAPPOINTED],
        "suggested_agent": _suggest_agent_for_emotion(emotion),
    }


def _suggest_agent_for_emotion(emotion: Emotion) -> Optional[str]:
    """
    Suggest an agent based on detected emotion.
    """
    emotion_agent_map = {
        Emotion.CONFUSED: "explain",      # More patient explanations
        Emotion.FRUSTRATED: "debug",      # Focus on fixing issues
        Emotion.CURIOUS: "research",      # Deep dive into topics
        Emotion.ANXIOUS: "summary",       # Quick, concise answers
    }
    return emotion_agent_map.get(emotion)
