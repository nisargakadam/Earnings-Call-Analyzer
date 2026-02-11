#!/usr/bin/env python3
"""
Earnings Call Analyzer
Analyzes earnings call transcripts and generates trading recommendations.
Supports:
- Pattern-based “Anthropic” path (placeholder / non-API)
- Hugging Face sentiment integration via HuggingFaceAnalyzer
"""

import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from huggingface_integration import HuggingFaceAnalyzer


@dataclass
class EarningsMetrics:
    """Structure for extracted earnings metrics"""
    company: str
    ticker: str
    quarter: str
    year: int
    revenue: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    revenue_vs_estimate: Optional[str] = None
    eps_actual: Optional[float] = None
    eps_vs_estimate: Optional[str] = None
    eps_surprise_pct: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    guidance_direction: Optional[str] = None
    guidance_summary: Optional[str] = None
    management_sentiment: Optional[str] = None
    key_growth_drivers: Optional[str] = None
    recommendation: Optional[str] = None
    options_strategy: Optional[str] = None
    confidence: Optional[str] = None
    reasoning: Optional[str] = None


class EarningsCallAnalyzer:
    """Analyzes earnings call transcripts"""

    def __init__(self, use_huggingface: bool = False):
        """
        Args:
            use_huggingface: Whether to use Hugging Face models instead
        """
        self.use_huggingface = use_huggingface

    def extract_metadata(self, transcript: str, filename: str = "") -> Dict[str, str]:
        """Extract basic metadata from transcript + filename."""
        metadata: Dict[str, str] = {}

        # Try to extract from filename (e.g., "AMD_2017_Q4.txt")
        if filename:
            base = filename.replace(".txt", "")
            parts = base.split("_")
            if len(parts) >= 3:
                metadata["company"] = parts[0]
                metadata["year"] = parts[1]
                metadata["quarter"] = parts[2]

        # Extract from transcript header if available
        # Examples: "Fourth Quarter 2017" or "Q4 2017"
        quarter_match = re.search(
            r'(Q[1-4]|[Ff]irst|[Ss]econd|[Tt]hird|[Ff]ourth)\s+[Qq]uarter\s+(\d{4})',
            transcript[:800],
        )
        if quarter_match:
            quarter_map = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
            q = quarter_match.group(1)
            if q.lower() in quarter_map:
                q = quarter_map[q.lower()]
            metadata["quarter"] = q
            metadata["year"] = quarter_match.group(2)

        # If company not found, fall back to ticker-like token from transcript top
        if "company" not in metadata:
            m = re.search(r"\b([A-Z]{2,6})\b", transcript[:200])
            if m:
                metadata["company"] = m.group(1)

        return metadata

    def analyze_with_llm(self, transcript: str, metadata: Dict) -> EarningsMetrics:
        """
        This builds a prompt-like structure for consistency with your earlier code.
        We still do local extraction in both paths (pattern-based + HF sentiment).
        """
        prompt = f"""You are a financial analyst analyzing an earnings call transcript.

Company: {metadata.get('company', 'Unknown')}
Quarter: {metadata.get('quarter', 'Unknown')} {metadata.get('year', 'Unknown')}

TRANSCRIPT:
{transcript[:15000]}

Please analyze this earnings call."""
        if self.use_huggingface:
            return self._analyze_with_huggingface(prompt, metadata)
        return self._analyze_with_anthropic(prompt, metadata)

    def _extract_transcript_from_prompt(self, prompt: str) -> str:
        """Safely extract transcript text from the prompt block."""
        if "TRANSCRIPT:" not in prompt:
            return prompt
        after = prompt.split("TRANSCRIPT:", 1)[1]
        # Trim off anything after the transcript section marker if present
        if "Please analyze" in after:
            after = after.split("Please analyze", 1)[0]
        return after.strip()

    def _base_regex_extract(self, transcript: str, metadata: Dict) -> EarningsMetrics:
        """Shared regex extraction used by both paths."""
        metrics = EarningsMetrics(
            company=metadata.get("company", "Unknown"),
            ticker=metadata.get("company", "Unknown"),
            quarter=metadata.get("quarter", "Unknown"),
            year=int(metadata.get("year", 0) or 0),
        )

        # Revenue (billions)
        rev_match = re.search(r"revenue[^\d]*\$?(\d+\.?\d*)\s*(billion|B)\b", transcript, re.I)
        if rev_match:
            metrics.revenue = float(rev_match.group(1))

        # Revenue growth YoY (%)
        growth_match = re.search(r"(\d+\.?\d*)%?\s*(year[- ]over[- ]year|YoY|y/y)\b", transcript, re.I)
        if growth_match:
            metrics.revenue_growth_yoy = float(growth_match.group(1))

        # Margins
        gm_match = re.search(r"gross margin[^\d]*(\d+\.?\d*)%\b", transcript, re.I)
        if gm_match:
            metrics.gross_margin = float(gm_match.group(1))

        om_match = re.search(r"operating margin[^\d]*(\d+\.?\d*)%\b", transcript, re.I)
        if om_match:
            metrics.operating_margin = float(om_match.group(1))

        # Guidance direction keywords
        if re.search(r"(rais|increas|updat|higher).{0,40}guidance", transcript, re.I):
            metrics.guidance_direction = "raised"
        elif re.search(r"(lower|reduc|cut|decreas).{0,40}guidance", transcript, re.I):
            metrics.guidance_direction = "lowered"
        else:
            metrics.guidance_direction = "maintained"

        # Key products/drivers (very rough heuristic)
        product_keywords = re.findall(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:processor|GPU|CPU|product|platform)\b",
            transcript,
        )
        if product_keywords:
            metrics.key_growth_drivers = ", ".join(list(dict.fromkeys(product_keywords))[:3])

        return metrics

    def _score_and_recommend(self, metrics: EarningsMetrics) -> None:
        """Apply your scoring rules and fill recommendation fields in-place."""
        score = 0

        # Guidance (most important)
        if metrics.guidance_direction == "raised":
            score += 3
        elif metrics.guidance_direction == "lowered":
            score -= 3

        # Revenue growth
        if metrics.revenue_growth_yoy is not None and metrics.revenue_growth_yoy > 20:
            score += 2
        elif metrics.revenue_growth_yoy is not None and metrics.revenue_growth_yoy < 0:
            score -= 2

        # Sentiment
        if metrics.management_sentiment == "positive":
            score += 1
        elif metrics.management_sentiment == "negative":
            score -= 1

        # Margin (simple threshold)
        if metrics.gross_margin is not None and metrics.gross_margin > 30:
            score += 1

        # Recommendation buckets
        if score >= 3:
            metrics.recommendation = "BUY"
            metrics.options_strategy = "Buy calls - strong fundamentals"
            metrics.confidence = "High"
        elif score <= -3:
            metrics.recommendation = "SELL"
            metrics.options_strategy = "Buy puts or sell"
            metrics.confidence = "High"
        elif score > 0:
            metrics.recommendation = "BUY"
            metrics.options_strategy = "Moderate bullish position"
            metrics.confidence = "Medium"
        elif score < 0:
            metrics.recommendation = "SELL"
            metrics.options_strategy = "Protective puts"
            metrics.confidence = "Medium"
        else:
            metrics.recommendation = "HOLD"
            metrics.options_strategy = "Wait for clearer signals"
            metrics.confidence = "Low"

    def _analyze_with_anthropic(self, prompt: str, metadata: Dict) -> EarningsMetrics:
        """
        Pattern-based analyzer (acts like your old 'anthropic' path).
        No external API call here — it’s deterministic.
        """
        transcript = self._extract_transcript_from_prompt(prompt)
        metrics = self._base_regex_extract(transcript, metadata)

        # Simple word-count sentiment
        pos = len(re.findall(r"\b(strong|growth|beat|exceed|positive|outperform|momentum)\b", transcript, re.I))
        neg = len(re.findall(r"\b(weak|decline|miss|below|negative|underperform|concern)\b", transcript, re.I))
        if pos > neg * 1.5:
            metrics.management_sentiment = "positive"
        elif neg > pos * 1.5:
            metrics.management_sentiment = "negative"
        else:
            metrics.management_sentiment = "neutral"

        self._score_and_recommend(metrics)

        # Reasoning string
        reasons: List[str] = []
        if metrics.revenue_growth_yoy is not None:
            reasons.append(f"{metrics.revenue_growth_yoy}% revenue growth")
        if metrics.guidance_direction:
            reasons.append(f"guidance {metrics.guidance_direction}")
        if metrics.management_sentiment:
            reasons.append(f"{metrics.management_sentiment} sentiment")
        metrics.reasoning = "; ".join(reasons) if reasons else "Limited data available"

        return metrics

    def _analyze_with_huggingface(self, prompt: str, metadata: Dict) -> EarningsMetrics:
        """
        Hugging Face path:
        - Use regex extraction for metrics
        - Use HF model for sentiment
        - Apply same scoring rules
        """
        transcript = self._extract_transcript_from_prompt(prompt)
        metrics = self._base_regex_extract(transcript, metadata)

        hf = HuggingFaceAnalyzer()
        sent = hf.analyze_sentiment(transcript)

        # Expecting {"label": "positive"/"negative"/"neutral", "score": float}
        metrics.management_sentiment = sent.get("label")
        self._score_and_recommend(metrics)

        # Add concise reasoning
        label = sent.get("label", "unknown")
        score = sent.get("score", None)
        if score is not None:
            metrics.reasoning = f"HF sentiment={label} (score={score:.2f}); guidance={metrics.guidance_direction}"
        else:
            metrics.reasoning = f"HF sentiment={label}; guidance={metrics.guidance_direction}"

        return metrics

    def process_transcript(self, transcript_path: str) -> EarningsMetrics:
        """Process a single transcript file."""
        path = Path(transcript_path)
        transcript = path.read_text(encoding="utf-8", errors="ignore")
        metadata = self.extract_metadata(transcript, path.name)
        return self.analyze_with_llm(transcript, metadata)

    def process_directory(self, directory_path: str, output_csv: str = "earnings_analysis.csv") -> List[EarningsMetrics]:
        """Process all transcript files in a directory."""
        directory = Path(directory_path)
        results: List[EarningsMetrics] = []

        txt_files = sorted(directory.glob("*.txt"))
        print(f"Found {len(txt_files)} transcript file(s) in {directory}")

        for txt_file in txt_files:
            print(f"\nProcessing: {txt_file.name}")
            try:
                metrics = self.process_transcript(str(txt_file))
                results.append(metrics)
                print(f"  → {metrics.recommendation} ({metrics.confidence} confidence)")
            except Exception as e:
                print(f"  → Error: {e}")

        self.save_to_csv(results, output_csv)
        print(f"\n✓ Saved {len(results)} analyses to {output_csv}")
        return results

    def save_to_csv(self, results: List[EarningsMetrics], output_path: str) -> None:
        """Save analysis results to CSV."""
        if not results:
            print("No results to save")
            return

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze earnings call transcript(s).")
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to a .txt transcript file OR a directory containing .txt files",
    )
    parser.add_argument(
        "--use_huggingface",
        action="store_true",
        help="Use Hugging Face sentiment model (via HuggingFaceAnalyzer).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="earnings_analysis.csv",
        help="Output CSV path (used when processing a directory).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EARNINGS CALL ANALYZER")
    print("=" * 60)

    analyzer = EarningsCallAnalyzer(use_huggingface=args.use_huggingface)

    # If no path provided, try sensible defaults
    target = args.path.strip()
    if not target:
        # Common sandbox default
        if Path("/mnt/data/AMD_2017_Q4.txt").exists():
            target = "/mnt/data/AMD_2017_Q4.txt"
        else:
            target = "."

    p = Path(target)
    if p.is_file():
        metrics = analyzer.process_transcript(str(p))
        print("\n" + "=" * 60)
        print("ANALYSIS RESULT")
        print("=" * 60)
        print(f"{metrics.company} {metrics.quarter} {metrics.year}")
        print(f"Revenue: {metrics.revenue}B")
        print(f"Revenue Growth YoY: {metrics.revenue_growth_yoy}%")
        print(f"Guidance: {metrics.guidance_direction}")
        print(f"Sentiment: {metrics.management_sentiment}")
        print(f"→ {metrics.recommendation} ({metrics.confidence})")
        print(f"Strategy: {metrics.options_strategy}")
        print(f"Reason: {metrics.reasoning}")
    elif p.is_dir():
        analyzer.process_directory(str(p), args.out_csv)
    else:
        raise FileNotFoundError(f"Path not found: {target}")


if __name__ == "__main__":
    main()
