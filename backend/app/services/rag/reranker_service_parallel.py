"""
LLM-based Re-ranking Service with Parallel Processing
Uses business rules to improve retrieval accuracy for SEC filings
Supports parallel scoring for faster performance
"""

import logging
import re
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import ollama
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Business rules for SEC filing relevance scoring
RERANKER_SYSTEM_PROMPT = """You are a relevance scoring assistant for SEC filings. Score how well a document chunk answers a financial question.

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

Given the query and document chunk, return ONLY a decimal score between 0.0 and 1.0.
Examples: 0.0, 0.3, 0.7, 0.9, 1.0"""


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


class ParallelLLMReranker:
    """LLM-based document re-ranking service with parallel processing"""

    def __init__(self, model: str = "qwen2.5:0.5b", max_workers: int = 4):
        """
        Initialize reranker with specified model

        Args:
            model: Ollama model to use for scoring (default: qwen2.5:0.5b)
            max_workers: Maximum number of parallel scoring workers
        """
        self.model = model
        self.max_workers = max_workers
        self.query_parser = QueryParser()
        self.system_prompt = RERANKER_SYSTEM_PROMPT
        logger.info(f"Parallel LLM Reranker initialized with model: {model}, workers: {max_workers}")

    def score_candidate(
        self,
        query: str,
        chunk: Dict[str, Any],
        query_intent: Optional[QueryIntent] = None
    ) -> Tuple[float, str]:
        """
        Score a single candidate chunk for relevance

        Args:
            query: User's question
            chunk: Document chunk with text and metadata
            query_intent: Pre-parsed query intent (optional)

        Returns:
            Tuple of (score between 0.0-1.0, reasoning)
        """
        if not query_intent:
            query_intent = self.query_parser.parse(query)

        # Extract metadata - handle different field names
        metadata = chunk.get('metadata', {})

        # Handle different field names for document type
        doc_type = metadata.get('doc_type') or metadata.get('document_type', 'unknown')

        # Handle different field names for fiscal year
        fiscal_year = metadata.get('fiscal_year') or metadata.get('year')

        # Extract quarter from metadata or filename
        quarter = metadata.get('quarter')
        if not quarter and 'filename' in metadata:
            # Try to extract from filename (e.g., "10-Q_2024_Q2.txt")
            import re
            q_match = re.search(r'Q([1-4])', metadata['filename'])
            if q_match:
                quarter = f"Q{q_match.group(1)}"

        company = metadata.get('company') or metadata.get('ticker')

        # Build scoring prompt
        prompt = self._build_scoring_prompt(
            query, chunk['text'][:500], metadata, query_intent
        )

        try:
            # Call LLM for scoring
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                system=self.system_prompt,
                options={
                    'temperature': 0.0,  # Deterministic scoring
                    'num_predict': 10,    # Just need a score
                }
            )

            # Parse score from response
            score_text = response['response'].strip()
            score = self._parse_score(score_text)

            # Apply hard rules for document type mismatch
            if query_intent.period_type == 'annual' and '10-Q' in doc_type:
                score = min(score, 0.3)
                reasoning = f"10-Q doc for annual query (capped at 0.3)"
            elif query_intent.period_type == 'quarterly' and '10-K' in doc_type:
                score = min(score, 0.3)
                reasoning = f"10-K doc for quarterly query (capped at 0.3)"
            else:
                reasoning = f"Score: {score:.2f}"

            # Apply year matching boost/penalty
            if query_intent.fiscal_year and fiscal_year:
                try:
                    doc_year = int(fiscal_year) if isinstance(fiscal_year, (str, int)) else None
                    query_year = int(query_intent.fiscal_year)

                    if doc_year:
                        # For fiscal year queries, we need to check both the document year and year-1
                        # because fiscal 2024 data is in the 2025 10-K (filed after fiscal year end)
                        if query_intent.period_type == 'annual':
                            # For annual reports, the 10-K contains data for the previous year
                            # e.g., 10-K_2025 contains fiscal 2024 data
                            if doc_year == query_year + 1:  # Correct 10-K for the fiscal year
                                score = min(1.0, score * 1.5)  # Boost score by 50%
                                reasoning += f" | Year match boost (10-K_{doc_year} has FY{query_year} data)"
                            elif doc_year == query_year:  # Previous year's 10-K
                                score = min(1.0, score * 0.8)  # Slight penalty
                                reasoning += f" | Previous 10-K"
                            else:
                                score = min(score, 0.2)  # Wrong year
                                reasoning += f" | Year mismatch penalty"
                        else:
                            # For quarterly reports, year should match exactly
                            if doc_year == query_year:
                                score = min(1.0, score * 1.3)  # Boost for correct year
                                reasoning += f" | Year match boost"
                            else:
                                score = min(score, 0.2)  # Wrong year
                                reasoning += f" | Year mismatch penalty"
                except (ValueError, TypeError):
                    pass  # If year parsing fails, don't apply year rules

            return score, reasoning

        except Exception as e:
            logger.error(f"Error scoring candidate: {e}")
            # Return neutral score on error
            return 0.5, "Error in scoring"

    def _score_candidate_wrapper(self, args):
        """Wrapper for parallel processing"""
        query, chunk, query_intent = args
        score, reasoning = self.score_candidate(query, chunk, query_intent)
        return chunk, score, reasoning

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank all candidates and return top K using parallel processing

        Args:
            query: User's question
            candidates: List of candidate chunks from initial retrieval
            top_k: Number of top results to return

        Returns:
            Top K re-ranked candidates with scores
        """
        # Parse query intent once
        query_intent = self.query_parser.parse(query)

        logger.info(f"Parallel re-ranking {len(candidates)} candidates for query: {query[:50]}...")
        logger.info(f"Query intent: {query_intent}")
        logger.info(f"Using {self.max_workers} parallel workers")

        # Prepare arguments for parallel processing
        scoring_args = [(query, candidate, query_intent) for candidate in candidates]

        # Score all candidates in parallel
        scored_candidates = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scoring tasks
            future_to_candidate = {
                executor.submit(self._score_candidate_wrapper, args): args[1]
                for args in scoring_args
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_candidate):
                try:
                    candidate, score, reasoning = future.result(timeout=10)
                    candidate['rerank_score'] = score
                    candidate['rerank_reasoning'] = reasoning
                    scored_candidates.append(candidate)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Scoring timeout for candidate")
                    candidate = future_to_candidate[future]
                    candidate['rerank_score'] = 0.5
                    candidate['rerank_reasoning'] = "Timeout"
                    scored_candidates.append(candidate)
                except Exception as e:
                    logger.error(f"Error in parallel scoring: {e}")
                    candidate = future_to_candidate[future]
                    candidate['rerank_score'] = 0.5
                    candidate['rerank_reasoning'] = "Error"
                    scored_candidates.append(candidate)

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Log top scores
        for i, cand in enumerate(scored_candidates[:5]):
            metadata = cand.get('metadata', {})
            doc_type = metadata.get('doc_type') or metadata.get('document_type', 'unknown')
            year = metadata.get('fiscal_year') or metadata.get('year', 'unknown')

            # Extract quarter from metadata or filename
            quarter = metadata.get('quarter')
            if not quarter and 'filename' in metadata:
                import re
                q_match = re.search(r'Q([1-4])', metadata['filename'])
                if q_match:
                    quarter = f"Q{q_match.group(1)}"

            logger.info(
                f"  Rank {i+1}: Score={cand['rerank_score']:.2f}, "
                f"Type={doc_type}, "
                f"Year={year}, "
                f"Quarter={quarter or 'N/A'}"
            )

        # Return top K
        return scored_candidates[:top_k]

    def _build_scoring_prompt(
        self,
        query: str,
        text_excerpt: str,
        metadata: Dict[str, Any],
        query_intent: QueryIntent
    ) -> str:
        """Build prompt for LLM scoring"""

        return f"""Query: {query}

Query is asking for: {query_intent.period_type} data for fiscal year {query_intent.fiscal_year or 'unspecified'}

Document Metadata:
- Type: {metadata.get('doc_type') or metadata.get('document_type', 'unknown')}
- Fiscal Year: {metadata.get('fiscal_year') or metadata.get('year', 'unknown')}
- Quarter: {metadata.get('quarter', 'N/A')}
- Company: {metadata.get('company') or metadata.get('ticker', 'unknown')}

Text Excerpt:
{text_excerpt}

Based on the rules, what is the relevance score (0.0-1.0)?
Score:"""

    def _parse_score(self, score_text: str) -> float:
        """Parse score from LLM response"""

        # Extract first number from response
        import re
        match = re.search(r'(\d*\.?\d+)', score_text)
        if match:
            try:
                score = float(match.group(1))
                # Ensure score is between 0 and 1
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Default to 0.5 if parsing fails
        logger.warning(f"Could not parse score from: {score_text}")
        return 0.5


# Create singleton with parallel processing enabled
parallel_reranker = ParallelLLMReranker(max_workers=4)