import os
import re
import json
import logging

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_FALLBACK = {
    "article_summary": "AI analysis could not be completed. Please review manually.",
    "credibility_indicators": {
        "positive": [],
        "negative": ["AI reasoning failed — manual review needed"],
    },
    "risk_factors": ["Automated analysis unavailable"],
    "cross_source_verification": "Manual verification recommended.",
    "confidence_assessment": {
        "overall_verdict": "UNCERTAIN",
        "confidence_level": "LOW",
        "reasoning": "The AI reasoning engine could not produce a valid response.",
    },
    "supporting_sources": [],
    "ethical_disclaimer": (
        "This is an automated AI assessment. Always verify with trusted news sources "
        "before drawing any conclusions."
    ),
    "misinformation_warning": "",
}


class CredibilityReasoner:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def analyze(self, article_text: str, prediction: dict, retrieval: dict) -> dict:
        sources_text = "\n".join(
            f"- [{s.get('title', 'Source')}]: {s.get('snippet', '')}"
            for s in retrieval.get("sources", [])[:5]
        ) or "No external sources retrieved."

        prompt = f"""You are a fact-checking analyst. Analyze the article below and return a structured credibility assessment.

ARTICLE:
{article_text[:1800]}

ML CLASSIFIER:
Verdict: {prediction['label']} | Confidence: {prediction['confidence']}%
P(REAL)={prediction['proba_real']}%  P(FAKE)={prediction['proba_fake']}%

RETRIEVED SOURCES:
{sources_text}

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):

{{
  "article_summary": "<2-3 sentence neutral summary>",
  "credibility_indicators": {{
    "positive": ["<supporting element>"],
    "negative": ["<red flag>"]
  }},
  "risk_factors": ["<misinformation risk>"],
  "cross_source_verification": "<1-2 sentences on what the sources suggest>",
  "confidence_assessment": {{
    "overall_verdict": "<CREDIBLE | LIKELY FAKE | UNCERTAIN>",
    "confidence_level": "<HIGH | MEDIUM | LOW>",
    "reasoning": "<2-3 sentences explaining the judgment>"
  }},
  "supporting_sources": ["<source name from retrieved list>"],
  "ethical_disclaimer": "<brief disclaimer about AI limitations>",
  "misinformation_warning": "<specific warning if misinformation detected, else empty string>"
}}

Rules: base verdict on article + ML result; don't invent facts; if the article is too short, highly biased, or lacks verifiable claims, label as UNCERTAIN and specify "Incomplete/Unreliable Content" in the reasoning; keep lists to 2-4 items."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)

        except json.JSONDecodeError:
            logger.error("LLM returned non-JSON; using fallback")
            return _FALLBACK.copy()

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            result = _FALLBACK.copy()
            result["article_summary"] = f"Analysis failed: {e}"
            return result