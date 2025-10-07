import os
import boto3
from typing import List, Dict, Any
from aws_lambda_powertools import Logger
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from pydantic import ValidationError

from models.request import AnalysisRequest, Sentence
from services.clustering import ClusteringService


logger = Logger()
app = APIGatewayRestResolver()


class DuplicateIDError(ValueError):
    """Custom exception for duplicate ID validation errors."""
    pass

bedrock_region = os.getenv("BEDROCK_REGION", "us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name=bedrock_region)
clustering_service = ClusteringService(bedrock)


def deduplicate_sentences(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate sentence IDs, keeping first occurrence.
    
    Args:
        sentences: List of sentence dicts with 'id' field
        
    Returns:
        Deduplicated list of sentences
    """
    seen = set()
    deduplicated = []
    duplicates = []
    
    for sentence in sentences:
        sentence_id = sentence.get('id')
        if sentence_id not in seen:
            seen.add(sentence_id)
            deduplicated.append(sentence)
        else:
            duplicates.append(sentence_id)
    
    if duplicates:
        logger.warning(
            "Duplicate sentence IDs removed during deduplication",
            extra={"duplicate_ids": duplicates, "total_duplicates": len(duplicates)}
        )
    
    return deduplicated


@app.post("/analyze")
def analyze():
    """Validate request, run clustering, return JSON API response."""
    try:
        validation_mode = os.getenv("VALIDATION_MODE", "strict").lower()
        body = app.current_event.json_body
        
        # In dedup mode, remove duplicates before validation
        if validation_mode == "dedup":
            if body.get("baseline"):
                body["baseline"] = deduplicate_sentences(body["baseline"])
            if body.get("comparison"):
                body["comparison"] = deduplicate_sentences(body["comparison"])
        
        # Create and validate request
        request = AnalysisRequest(**body)
        
        # In strict mode, check for duplicates and raise error
        if validation_mode == "strict":
            baseline_dups, comparison_dups, overlap = request.get_duplicate_ids()
            
            if baseline_dups:
                raise DuplicateIDError(f"Duplicate sentence IDs found in baseline: {baseline_dups}")
            if comparison_dups:
                raise DuplicateIDError(f"Duplicate sentence IDs found in comparison: {comparison_dups}")
            if overlap:
                raise DuplicateIDError(f"Sentence IDs must be unique across baseline and comparison: {overlap}")
        
        # Run clustering analysis
        result = clustering_service.analyze(request)

        return {"statusCode": 200, "body": result.model_dump()}

    except ValidationError as e:
        logger.warning("Validation failed", extra={"error": str(e)})
        return {"statusCode": 400, "body": {"error": "Invalid request", "details": str(e)}}
    
    except DuplicateIDError as e:
        # Duplicate ID errors from strict mode - client error
        logger.warning("Duplicate ID validation failed", extra={"error": str(e)})
        return {"statusCode": 400, "body": {"error": "Invalid request", "details": str(e)}}

    except Exception as e:
        # Includes ValueError from LLM output validation - server error
        logger.exception("Unhandled error", extra={"error": str(e)})
        return {"statusCode": 500, "body": {"error": "Internal error"}}


def lambda_handler(event, context):
    """Lambda entrypoint for API Gateway REST events."""
    return app.resolve(event, context)