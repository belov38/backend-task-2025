import json
import os
import re
import time
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from aws_lambda_powertools import Logger
from models.response import AnalysisResponse, Cluster
from models.request import AnalysisRequest, Sentence

logger = Logger(child=True)


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

        if len(request.baseline) <= self.chunk_size:
            return self._single_cluster(request)

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

        return AnalysisResponse(clusters=merged_clusters)
    
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

{sentences_text}

{{"clusters":[{{"title":"Name","sentiment":"neg","sentences":["id"],"keyInsights":["Brief"]}}]}}

ONLY JSON ABOVE. NO OTHER TEXT."""
        
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
            return {"clusters": []}
    
    def _single_cluster(self, request) -> AnalysisResponse:
        """Single model call for small datasets."""
        result = self._cluster_chunk(request.baseline, request.theme, 0)
        clusters = [Cluster(**c) for c in result.get("clusters", [])]
        return AnalysisResponse(clusters=clusters)
    
    def _merge_clusters(self, chunk_results: List[Dict[str, Any]]) -> List[Cluster]:
        """Merge sub-clusters by title and sentiment, deduplicating insights."""
        all_clusters: List[Dict[str, Any]] = []
        for result in chunk_results:
            all_clusters.extend(result.get("clusters", []))

        if not all_clusters:
            return []

        merged_map: Dict[str, Dict[str, Any]] = {}

        for cluster in all_clusters:
            title_key = cluster["title"].lower().strip()

            if title_key not in merged_map:
                merged_map[title_key] = {
                    "title": cluster["title"],
                    "sentiment": cluster["sentiment"],
                    "sentences": list(cluster["sentences"]),
                    "keyInsights": list(cluster["keyInsights"]),
                }
            else:
                merged_map[title_key]["sentences"].extend(cluster["sentences"])

                for insight in cluster["keyInsights"]:
                    if insight not in merged_map[title_key]["keyInsights"]:
                        merged_map[title_key]["keyInsights"].append(insight)

                if cluster["sentiment"] == "negative":
                    merged_map[title_key]["sentiment"] = "negative"

        return [Cluster(**data) for data in merged_map.values()]
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from the model response and parse into a dict.

        Tolerates optional markdown code fences and extracts the top-level
        object containing the "clusters" key.
        """
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        match = re.search(r'\{.*"clusters".*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning(
                "JSON parse failed",
                extra={"error": str(e), "sample": text[:200]},
            )
            return {"clusters": []}