"""
LLM-based Re-ranking Service with Batch Processing
Scores all candidates in a single LLM call for better efficiency
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import ollama
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Business rules for SEC filing relevance scoring
BATCH_RERANKER_SYSTEM_PROMPT = """You are a relevance scoring assistant for SEC filings. Score how well document chunks answer a financial question.

CRITICAL RULES:

1. Annual vs Quarterly Documents:
   - Questions about "fiscal year", "fiscal 20XX", "FY 20XX", "annual", "full year" → REQUIRE 10-K documents
   - Questions about "Q1/Q2/Q3/Q4", "quarter", "three months", "quarterly" → REQUIRE 10-Q documents
   - 10-K is the ONLY authoritative source for annual totals
   - 10-Q is the ONLY authoritative source for quarterly data

2. Document Type Scoring:
   - If asking for annual data but chunk is from 10-Q: maximum score 0.3
   - If asking for quarterly data but chunk is from 10-K: maximum score 0.3
   - Correct document type can score up to 1.0

3. Fiscal Year Matching:
   - Exact fiscal year match: can score 0.7-1.0
   - Wrong fiscal year: maximum score 0.2
   - For quarters: both year AND quarter must match

4. Company Matching:
   - Must match the requested company ticker
   - Wrong company: score 0.0

Return a JSON array with scores for each document in order.
Example: [0.9, 0.3, 0.7, 0.0, 0.5]"""


@dataclass
class QueryIntent:
    """Parsed intent from user query"""
    period_type: str  # 'annual' or 'quarterly'
    fiscal_year: Optional[int] = None
    quarter: Optional[str] = None
    company: Optional[str] = None


class QueryParser:
    """Parse financial queries to extract intent"""

    def parse(self, query: str, default_company: str = "AAPL") -> QueryIntent:
        """
        Extract intent from query

        Args:
            query: User's question
            default_company: Default company if not specified

        Returns:
            QueryIntent with parsed information
        """
        query_lower = query.lower()

        # Detect period type
        period_type = self._detect_period_type(query_lower)

        # Extract fiscal year
        fiscal_year = self._extract_fiscal_year(query_lower)

        # Extract quarter if quarterly
        quarter = None
        if period_type == 'quarterly':
            quarter = self._extract_quarter(query_lower)

        # Extract company (default to AAPL for now)
        company = self._extract_company(query_lower) or default_company

        return QueryIntent(
            period_type=period_type,
            fiscal_year=fiscal_year,
            quarter=quarter,
            company=company
        )

    def _detect_period_type(self, query: str) -> str:
        """Detect if query is about annual or quarterly data"""

        # Annual indicators
        annual_patterns = [
            'fiscal year', 'fiscal 20', 'fy 20', 'fy20',
            'annual', 'full year', 'for the year',
            'total net sales', 'total revenue',
            'ended september'  # Apple's fiscal year end
        ]

        # Quarterly indicators
        quarterly_patterns = [
            r'\bq[1-4]\b', 'quarter', 'quarterly',
            'three months', '3 months', 'three-month',
            'ended march', 'ended june', 'ended december'
        ]

        # Check for quarterly patterns first (more specific)
        for pattern in quarterly_patterns:
            if re.search(pattern, query):
                return 'quarterly'

        # Check for annual patterns
        for pattern in annual_patterns:
            if pattern in query:
                return 'annual'

        # Default to annual if unclear
        return 'annual'

    def _extract_fiscal_year(self, query: str) -> Optional[int]:
        """Extract fiscal year from query"""

        # Look for 4-digit years
        year_matches = re.findall(r'\b20\d{2}\b', query)
        if year_matches:
            # Return the most recent year mentioned
            return max(int(year) for year in year_matches)

        # Look for 2-digit fiscal year references (FY25, fiscal 25)
        fy_matches = re.findall(r'(?:fy|fiscal)\s*(\d{2})\b', query)
        if fy_matches:
            year = int(fy_matches[-1])
            # Convert 2-digit to 4-digit (assume 2000s)
            return 2000 + year if year < 50 else 1900 + year

        return None

    def _extract_quarter(self, query: str) -> Optional[str]:
        """Extract quarter from query"""

        # Direct quarter references (Q1, Q2, Q3, Q4)
        quarter_match = re.search(r'\bq([1-4])\b', query)
        if quarter_match:
            return f"Q{quarter_match.group(1)}"

        # Month-based quarter detection
        if 'first quarter' in query or 'ended december' in query:
            return 'Q1'
        elif 'second quarter' in query or 'ended march' in query:
            return 'Q2'
        elif 'third quarter' in query or 'ended june' in query:
            return 'Q3'
        elif 'fourth quarter' in query or 'ended september' in query:
            return 'Q4'

        return None

    def _extract_company(self, query: str) -> Optional[str]:
        """Extract company ticker from query"""

        # Common company name to ticker mapping
        company_map = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'amazon': 'AMZN',
            'tesla': 'TSLA'
        }

        for company_name, ticker in company_map.items():
            if company_name in query:
                return ticker

        # Look for direct ticker mentions
        ticker_match = re.search(r'\b([A-Z]{2,5})\b', query.upper())
        if ticker_match and ticker_match.group(1) in company_map.values():
            return ticker_match.group(1)

        return None


class BatchLLMReranker:
    """LLM-based document re-ranking service with batch scoring"""

    def __init__(self, model: str = "qwen2.5:0.5b"):
        """
        Initialize reranker with specified model

        Args:
            model: Ollama model to use for scoring
        """
        self.model = model
        self.query_parser = QueryParser()
        self.system_prompt = BATCH_RERANKER_SYSTEM_PROMPT
        logger.info(f"Batch LLM Reranker initialized with model: {model}")

    def score_all_candidates(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        query_intent: Optional[QueryIntent] = None
    ) -> List[Tuple[float, str]]:
        """
        Score all candidate chunks in a single LLM call

        Args:
            query: User's question
            chunks: List of document chunks with text and metadata
            query_intent: Pre-parsed query intent (optional)

        Returns:
            List of tuples (score, reasoning) for each chunk
        """
        if not query_intent:
            query_intent = self.query_parser.parse(query)

        # Build batch scoring prompt
        prompt = self._build_batch_scoring_prompt(query, chunks, query_intent)

        try:
            # Call LLM once for all candidates
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                system=self.system_prompt,
                options={
                    'temperature': 0.0,  # Deterministic scoring
                    'num_predict': 100,   # Enough for array of scores
                }
            )

            # Parse scores from response
            scores_text = response['response'].strip()
            raw_scores = self._parse_batch_scores(scores_text, len(chunks))

            # Apply business rules to each score
            final_scores = []
            for i, chunk in enumerate(chunks):
                score = raw_scores[i] if i < len(raw_scores) else 0.5
                score, reasoning = self._apply_business_rules(
                    score, chunk, query_intent
                )
                final_scores.append((score, reasoning))

            return final_scores

        except Exception as e:
            logger.error(f"Error in batch scoring: {e}")
            # Return neutral scores on error
            return [(0.5, "Error in scoring")] * len(chunks)

    def _apply_business_rules(
        self,
        base_score: float,
        chunk: Dict[str, Any],
        query_intent: QueryIntent
    ) -> Tuple[float, str]:
        """Apply business rules to adjust the base score"""

        score = base_score
        reasoning = f"Base score: {base_score:.2f}"

        metadata = chunk.get('metadata', {})
        doc_type = metadata.get('doc_type') or metadata.get('document_type', 'unknown')
        fiscal_year = metadata.get('fiscal_year') or metadata.get('year')

        # Apply hard rules for document type mismatch
        if query_intent.period_type == 'annual' and '10-Q' in doc_type:
            score = min(score, 0.3)
            reasoning = f"10-Q doc for annual query (capped at 0.3)"
        elif query_intent.period_type == 'quarterly' and '10-K' in doc_type:
            score = min(score, 0.3)
            reasoning = f"10-K doc for quarterly query (capped at 0.3)"

        # Apply year matching boost/penalty
        if query_intent.fiscal_year and fiscal_year:
            try:
                doc_year = int(fiscal_year) if isinstance(fiscal_year, (str, int)) else None
                query_year = int(query_intent.fiscal_year)

                if doc_year:
                    if query_intent.period_type == 'annual':
                        # 10-K contains previous year's data
                        if doc_year == query_year + 1:  # Correct 10-K
                            score = min(1.0, score * 1.5)
                            reasoning += f" | Year match boost"
                        elif doc_year == query_year:  # Previous year's 10-K
                            score = min(1.0, score * 0.8)
                            reasoning += f" | Previous 10-K"
                        else:
                            score = min(score, 0.2)
                            reasoning += f" | Year mismatch"
                    else:
                        # For quarterly reports
                        if doc_year == query_year:
                            score = min(1.0, score * 1.3)
                            reasoning += f" | Year match boost"
                        else:
                            score = min(score, 0.2)
                            reasoning += f" | Year mismatch"
            except (ValueError, TypeError):
                pass

        return score, reasoning

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank all candidates and return top K using batch scoring

        Args:
            query: User's question
            candidates: List of candidate chunks from initial retrieval
            top_k: Number of top results to return

        Returns:
            Top K re-ranked candidates with scores
        """
        # Parse query intent once
        query_intent = self.query_parser.parse(query)

        logger.info(f"Batch re-ranking {len(candidates)} candidates for query: {query[:50]}...")
        logger.info(f"Query intent: {query_intent}")

        # Score all candidates in one LLM call
        scores_and_reasoning = self.score_all_candidates(query, candidates, query_intent)

        # Apply scores to candidates
        for i, candidate in enumerate(candidates):
            if i < len(scores_and_reasoning):
                score, reasoning = scores_and_reasoning[i]
                candidate['rerank_score'] = score
                candidate['rerank_reasoning'] = reasoning
            else:
                candidate['rerank_score'] = 0.5
                candidate['rerank_reasoning'] = "Not scored"

        # Sort by score descending
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Log top scores
        for i, cand in enumerate(candidates[:5]):
            metadata = cand.get('metadata', {})
            doc_type = metadata.get('doc_type') or metadata.get('document_type', 'unknown')
            year = metadata.get('fiscal_year') or metadata.get('year', 'unknown')

            logger.info(
                f"  Rank {i+1}: Score={cand['rerank_score']:.2f}, "
                f"Type={doc_type}, Year={year}, "
                f"Reason={cand['rerank_reasoning']}"
            )

        # Return top K
        return candidates[:top_k]

    def _build_batch_scoring_prompt(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        query_intent: QueryIntent
    ) -> str:
        """Build prompt for batch LLM scoring"""

        documents_text = ""
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            text_excerpt = chunk['text'][:500]  # Keep full 500 chars for better context

            documents_text += f"""
Document {i}:
- Type: {metadata.get('doc_type') or metadata.get('document_type', 'unknown')}
- Fiscal Year: {metadata.get('fiscal_year') or metadata.get('year', 'unknown')}
- Quarter: {metadata.get('quarter', 'N/A')}
- Company: {metadata.get('company') or metadata.get('ticker', 'unknown')}
Text: {text_excerpt}
"""

        return f"""Query: {query}

Query is asking for: {query_intent.period_type} data for fiscal year {query_intent.fiscal_year or 'unspecified'}

Score each of the following {len(chunks)} documents for relevance (0.0-1.0):
{documents_text}

Return ONLY a JSON array of scores in order. Example: [0.9, 0.3, 0.7, 0.0, 0.5]
Scores:"""

    def _parse_batch_scores(self, scores_text: str, expected_count: int) -> List[float]:
        """Parse array of scores from LLM response"""

        try:
            # Try to parse as JSON array
            scores_text = scores_text.strip()

            # Find JSON array in response
            import re
            array_match = re.search(r'\[[\d.,\s]+\]', scores_text)
            if array_match:
                scores_str = array_match.group(0)
                scores = json.loads(scores_str)
                return [float(s) for s in scores]
        except:
            pass

        # Fallback: extract all numbers
        try:
            import re
            numbers = re.findall(r'(\d*\.?\d+)', scores_text)
            scores = [float(n) for n in numbers[:expected_count]]

            # Pad with 0.5 if not enough scores
            while len(scores) < expected_count:
                scores.append(0.5)

            return scores
        except:
            logger.warning(f"Could not parse batch scores from: {scores_text}")
            return [0.5] * expected_count


# Create singleton instance
batch_reranker = BatchLLMReranker(model="qwen2.5:0.5b")