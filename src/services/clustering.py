import json
import os
import re
import time
from typing import Dict, List, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from aws_lambda_powertools import Logger
from models.response import AnalysisResponse, Cluster
from models.request import AnalysisRequest, Sentence

logger = Logger(child=True)

# Constants
CLUSTERS_KEY = "clusters"
TITLE_KEY = "title"
SENTIMENT_KEY = "sentiment"
SENTENCES_KEY = "sentences"
KEY_INSIGHTS_KEY = "keyInsights"
SENTIMENT_NEGATIVE = "negative"
SENTIMENT_POSITIVE = "positive"
SENTIMENT_NEUTRAL = "neutral"


class ClusteringService:
    """Service for clustering survey sentences using Bedrock Nova models.

    Environment variables:
    - MODEL_ID: Bedrock model identifier. Defaults to 'amazon.nova-lite-v1:0'.
    - CHUNK_SIZE: Number of sentences per chunk. Defaults to 50.
    - MAX_WORKERS: Max threads for parallel chunk processing. Defaults to 8.
    - BEDROCK_REGION: AWS region for Bedrock client. If not set, relies on client config.
    - MAX_TOKENS: Max tokens for model output. Defaults to 1500.
    - TEMPERATURE: Sampling temperature. Defaults to 0.
    - STOP_SEQUENCE: Optional stop sequence, default to '}]}'.
    """

    def __init__(self, bedrock_client):
        self.bedrock = bedrock_client
        self.model_id = os.getenv("MODEL_ID", "amazon.nova-lite-v1:0")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "50"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "8"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1500"))
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.stop_sequence = os.getenv("STOP_SEQUENCE", "}]}")
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Analyze the request and return merged clusters.

        For small datasets (<= CHUNK_SIZE), runs a single model call. For larger
        datasets, splits into chunks and processes in parallel, then merges.
        """
        total_start = time.time()

        # Build set of valid input sentence IDs for validation
        valid_ids = {s.id for s in request.baseline}
        
        # Get LLM output validation mode from env
        llm_validation_mode = os.getenv("LLM_OUTPUT_VALIDATION", "strict").lower()

        if len(request.baseline) <= self.chunk_size:
            response = self._single_cluster(request)
            response = self._validate_and_filter_output_sentence_ids(response, valid_ids, llm_validation_mode)
            return response

        chunks = self._chunk_sentences(request.baseline)
        logger.append_keys(total_sentences=len(request.baseline), total_chunks=len(chunks))
        logger.info("Starting parallel clustering")

        parallel_start = time.time()

        chunk_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._cluster_chunk, chunk, request.theme, idx): idx
                for idx, chunk in enumerate(chunks)
            }

            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    chunk_results.append(result)
                except Exception as e:
                    logger.exception("Chunk processing failed", extra={"chunk_idx": chunk_idx, "error": str(e)})

        parallel_time = time.time() - parallel_start

        merge_start = time.time()
        merged_clusters = self._merge_clusters(chunk_results)
        merge_time = time.time() - merge_start

        total_time = time.time() - total_start

        logger.info(
            "Clustering timing",
            extra={
                "parallel_seconds": round(parallel_time, 2),
                "merge_seconds": round(merge_time, 3),
                "total_seconds": round(total_time, 2),
            },
        )

        response = AnalysisResponse(clusters=merged_clusters)
        response = self._validate_and_filter_output_sentence_ids(response, valid_ids, llm_validation_mode)
        return response
    
    def _chunk_sentences(self, sentences: List[Sentence]) -> List[List[Sentence]]:
        """Split sentences into equal-sized chunks respecting CHUNK_SIZE."""
        chunks: List[List[Any]] = []
        for i in range(0, len(sentences), self.chunk_size):
            chunks.append(sentences[i : i + self.chunk_size])
        return chunks
    
    def _cluster_chunk(self, sentences: List[Sentence], theme: str, chunk_idx: int) -> Dict[str, Any]:
        chunk_start = time.time()
        
        sentences_text = "\n".join([
            f"[{s.id}] {s.sentence[:100]}" for s in sentences
        ])
        
        prompt = f"""CRITICAL: Return ONLY valid JSON. No explanations.

Cluster {len(sentences)} sentences. Max 2 groups.

Rules:
- Sentiment must be one of: "positive", "negative", "neutral".
- "sentences" must be a list of sentence ids exactly as provided in brackets.

Input:
{sentences_text}

Response JSON example (structure only, replace values):
{{"clusters":[{{"title":"Name","sentiment":"negative","sentences":["id"],"keyInsights":["Brief"]}}]}}

ONLY RETURN THE JSON OBJECT. NO OTHER TEXT."""
        
        try:
            bedrock_start = time.time()

            response = self.bedrock.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stopSequences": [self.stop_sequence] if self.stop_sequence else [],
                },
            )

            bedrock_time = time.time() - bedrock_start

            text = response["output"]["message"]["content"][0]["text"]

            parse_start = time.time()
            result = self._parse_json(text)
            parse_time = time.time() - parse_start

            total_time = time.time() - chunk_start

            logger.info(
                "Chunk processed",
                extra={
                    "chunk_idx": chunk_idx,
                    "sentences": len(sentences),
                    "bedrock_seconds": round(bedrock_time, 2),
                    "parse_seconds": round(parse_time, 3),
                    "total_seconds": round(total_time, 2),
                },
            )

            return result

        except Exception as e:
            logger.exception(
                "Chunk failed",
                extra={
                    "chunk_idx": chunk_idx,
                    "elapsed_seconds": round(time.time() - chunk_start, 2),
                    "error": str(e),
                },
            )
            return {CLUSTERS_KEY: []}
    
    def _single_cluster(self, request) -> AnalysisResponse:
        """Single model call for small datasets."""
        result = self._cluster_chunk(request.baseline, request.theme, 0)
        clusters = [Cluster(**c) for c in result.get(CLUSTERS_KEY, [])]
        return AnalysisResponse(clusters=clusters)
    
    @staticmethod
    def _normalize_to_string_list(value: Any) -> List[str]:
        """Convert a value to a list of strings, handling edge cases.
        
        Args:
            value: Can be a string, list, or other type
            
        Returns:
            List of strings. Empty list if value is invalid.
        """
        if isinstance(value, str):
            return [value]
        elif isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        else:
            return []
    
    def _merge_clusters(self, chunk_results: List[Dict[str, Any]]) -> List[Cluster]:
        """Merge sub-clusters by title, deduplicating insights and sentences.
        
        Clusters with the same title (case-insensitive) are merged together.
        Sentiment priority: negative > positive (most conservative approach).
        """
        all_clusters: List[Dict[str, Any]] = []
        for result in chunk_results:
            all_clusters.extend(result.get(CLUSTERS_KEY, []))

        if not all_clusters:
            return []

        merged_map: Dict[str, Dict[str, Any]] = {}

        for cluster in all_clusters:
            title_key = cluster[TITLE_KEY].lower().strip()

            if title_key not in merged_map:
                # Initialize new cluster entry
                merged_map[title_key] = {
                    TITLE_KEY: cluster[TITLE_KEY],
                    SENTIMENT_KEY: cluster[SENTIMENT_KEY],
                    SENTENCES_KEY: list(cluster[SENTENCES_KEY]),
                    KEY_INSIGHTS_KEY: list(cluster[KEY_INSIGHTS_KEY]),
                }
            else:
                # Merge into existing cluster
                existing = merged_map[title_key]
                existing[SENTENCES_KEY].extend(cluster[SENTENCES_KEY])
                
                # Add unique insights only
                for insight in cluster[KEY_INSIGHTS_KEY]:
                    if insight not in existing[KEY_INSIGHTS_KEY]:
                        existing[KEY_INSIGHTS_KEY].append(insight)
                
                # Negative sentiment takes precedence (conservative approach)
                if cluster[SENTIMENT_KEY] == SENTIMENT_NEGATIVE:
                    existing[SENTIMENT_KEY] = SENTIMENT_NEGATIVE

        return [Cluster(**data) for data in merged_map.values()]
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and normalize JSON from the model response.

        Handles markdown code fences, extracts the JSON object, and normalizes
        field types to ensure consistency.
        
        Returns:
            Dict with 'clusters' key containing list of normalized cluster dicts.
            Returns empty clusters list on parse failure.
        """
        # Remove markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        
        # Extract JSON object containing clusters
        match = re.search(r'\{.*"clusters".*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        try:
            data: Dict[str, Any] = json.loads(text.strip())
            clusters = data.get(CLUSTERS_KEY)
            
            if not isinstance(clusters, list):
                return {CLUSTERS_KEY: []}

            normalized: List[Dict[str, Any]] = []
            for cluster in clusters:
                normalized_cluster = self._normalize_cluster(cluster)
                if normalized_cluster:
                    normalized.append(normalized_cluster)

            return {CLUSTERS_KEY: normalized}
            
        except json.JSONDecodeError as e:
            logger.warning(
                "JSON parse failed",
                extra={"error": str(e), "sample": text[:200]},
            )
            return {CLUSTERS_KEY: []}
    
    def _normalize_cluster(self, cluster: Any) -> Union[Dict[str, Any], None]:
        """Normalize a single cluster dict, ensuring correct field types.
        
        Args:
            cluster: Raw cluster data from model response
            
        Returns:
            Normalized cluster dict or None if invalid
        """
        if not isinstance(cluster, dict):
            return None
        
        title = cluster.get(TITLE_KEY)
        sentiment = cluster.get(SENTIMENT_KEY)
        
        # Both title and sentiment must be strings
        if not isinstance(title, str) or not isinstance(sentiment, str):
            return None
        
        sentences = self._normalize_to_string_list(cluster.get(SENTENCES_KEY, []))
        key_insights = self._normalize_to_string_list(cluster.get(KEY_INSIGHTS_KEY, []))
        normalized_sentiment = self._normalize_sentiment(sentiment)
        
        return {
            TITLE_KEY: title,
            SENTIMENT_KEY: normalized_sentiment,
            SENTENCES_KEY: sentences,
            KEY_INSIGHTS_KEY: key_insights,
        }

    @staticmethod
    def _normalize_sentiment(value: str) -> str:
        """Map arbitrary sentiment strings to allowed values.

        Allowed output values: "positive", "negative", "neutral".
        Unknown values default to "neutral".
        """
        if not isinstance(value, str):
            return SENTIMENT_NEUTRAL

        normalized = value.strip().lower()

        negative_aliases = {
            "neg", "negative", "bad", "poor", "unfavorable", "unfavourable",
            "sad", "angry", "detrimental", "unhappy"
        }
        positive_aliases = {
            "pos", "positive", "good", "great", "excellent", "favorable", "favourable",
            "happy", "satisfied"
        }
        neutral_aliases = {"neutral", "mixed", "mixed/neutral", "mixed neutral", "okay", "ok"}

        if normalized in negative_aliases:
            return SENTIMENT_NEGATIVE
        if normalized in positive_aliases:
            return SENTIMENT_POSITIVE
        if normalized in neutral_aliases:
            return SENTIMENT_NEUTRAL

        return SENTIMENT_NEUTRAL

    @staticmethod
    def _validate_and_filter_output_sentence_ids(response: AnalysisResponse, valid_ids: set, mode: str = 'strict') -> AnalysisResponse:
        """Validate and optionally filter sentence IDs in the response.

        Args:
            response: The analysis response to validate
            valid_ids: Set of valid sentence IDs from the input
            mode: 'strict' (raise error) or 'soft' (filter invalid IDs)

        Returns:
            AnalysisResponse with filtered clusters (in soft mode) and unprocessedSentences

        Raises:
            ValueError: If any sentence ID is invalid (strict mode only)
        """
        invalid_ids = []
        processed_ids = set()
        
        # Collect all sentence IDs from clusters and find invalid ones
        for cluster in response.clusters:
            for sentence_id in cluster.sentences:
                if sentence_id not in valid_ids:
                    invalid_ids.append(sentence_id)
                else:
                    processed_ids.add(sentence_id)
        
        if invalid_ids:
            logger.warning(
                "LLM output contains invalid sentence IDs",
                extra={
                    "invalid_ids": list(set(invalid_ids))[:10],
                    "total_invalid": len(set(invalid_ids)),
                    "mode": mode
                }
            )
            
            if mode == 'strict':
                raise ValueError(
                    f"LLM returned sentence IDs not present in input. "
                    f"Found {len(set(invalid_ids))} invalid IDs. "
                    f"Sample: {list(set(invalid_ids))[:5]}"
                )
            
            # Soft mode: filter out invalid IDs from clusters
            for cluster in response.clusters:
                cluster.sentences = [sid for sid in cluster.sentences if sid in valid_ids]
        
        # Find unprocessed sentences
        unprocessed = valid_ids - processed_ids
        if unprocessed:
            logger.info(
                "Some sentences were not processed by LLM",
                extra={
                    "unprocessed_count": len(unprocessed),
                    "total_input": len(valid_ids),
                    "coverage_percent": round((len(processed_ids) / len(valid_ids)) * 100, 2)
                }
            )
            response.unprocessedSentences = sorted(list(unprocessed))
        
        return response