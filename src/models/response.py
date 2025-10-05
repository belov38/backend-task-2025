from typing import List

from pydantic import BaseModel


class Cluster(BaseModel):
    """Cluster with metadata and sentence IDs."""

    title: str
    sentiment: str
    sentences: List[str]
    keyInsights: List[str]


class AnalysisResponse(BaseModel):
    """Top-level response containing identified clusters."""

    clusters: List[Cluster]