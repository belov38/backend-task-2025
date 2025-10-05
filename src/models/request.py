from typing import List, Optional

from pydantic import BaseModel


class Sentence(BaseModel):
    """A single survey sentence with an identifier."""

    sentence: str
    id: str


class AnalysisRequest(BaseModel):
    """Incoming request schema for clustering analysis."""

    surveyTitle: str
    theme: str
    baseline: List[Sentence]
    comparison: Optional[List[Sentence]] = None