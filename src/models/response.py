from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class Cluster(BaseModel):
    """Cluster with metadata and sentence IDs."""

    title: str = Field(..., min_length=1, max_length=500, description="Cluster title")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="Cluster sentiment")
    sentences: List[str] = Field(..., min_length=1, description="List of sentence IDs in this cluster")
    keyInsights: List[str] = Field(..., min_length=1, max_length=10, description="Key insights for this cluster")

    @field_validator('title')
    @classmethod
    def title_not_empty_or_whitespace(cls, v: str) -> str:
        """Validate title is not just whitespace."""
        if not v or not v.strip():
            raise ValueError('Title cannot be empty or only whitespace')
        return v.strip()

    @field_validator('sentences')
    @classmethod
    def sentences_not_empty(cls, v: List[str]) -> List[str]:
        """Validate sentences list contains valid IDs."""
        if not v:
            raise ValueError('Sentences list cannot be empty')
        # Check for empty or whitespace-only IDs
        for sentence_id in v:
            if not sentence_id or not sentence_id.strip():
                raise ValueError('Sentence IDs cannot be empty or only whitespace')
        return [s.strip() for s in v]

    @field_validator('keyInsights')
    @classmethod
    def key_insights_not_empty(cls, v: List[str]) -> List[str]:
        """Validate keyInsights contains non-empty strings."""
        if not v:
            raise ValueError('Key insights list cannot be empty')
        for insight in v:
            if not insight or not insight.strip():
                raise ValueError('Key insights cannot be empty or only whitespace')
        return [i.strip() for i in v]


class AnalysisResponse(BaseModel):
    """Top-level response containing identified clusters.
    
    Note: Allows soft clustering - same sentence ID can appear in multiple clusters.
    """

    clusters: List[Cluster] = Field(..., description="List of identified clusters")
    unprocessedSentences: List[str] = Field(
        default_factory=list,
        description="Sentence IDs that were not processed by LLM (for retry logic)"
    )