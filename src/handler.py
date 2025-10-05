import os
import boto3
from aws_lambda_powertools import Logger
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from pydantic import ValidationError

from models.request import AnalysisRequest
from services.clustering import ClusteringService


logger = Logger()
app = APIGatewayRestResolver()

# Warm start resources
bedrock_region = os.getenv("BEDROCK_REGION", "us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name=bedrock_region)
clustering_service = ClusteringService(bedrock)


@app.post("/analyze")
def analyze():
    """Validate request, run clustering, return JSON API response."""
    try:
        request = AnalysisRequest(**app.current_event.json_body)

        result = clustering_service.analyze(request)

        return {"statusCode": 200, "body": result.model_dump()}

    except ValidationError as e:
        logger.warning("Validation failed", extra={"error": str(e)})
        return {"statusCode": 400, "body": {"error": "Invalid request", "details": str(e)}}

    except Exception as e:
        logger.exception("Unhandled error", extra={"error": str(e)})
        return {"statusCode": 500, "body": {"error": "Internal error"}}


def lambda_handler(event, context):
    """Lambda entrypoint for API Gateway REST events."""
    return app.resolve(event, context)