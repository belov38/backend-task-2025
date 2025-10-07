from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Sentence(BaseModel):
    """A single survey sentence with an identifier."""

    sentence: str = Field(..., min_length=1, max_length=10000, description="The sentence text")
    id: str = Field(..., min_length=1, max_length=500, description="Unique sentence identifier")

    @field_validator('sentence')
    @classmethod
    def sentence_not_empty_or_whitespace(cls, v: str) -> str:
        """Validate sentence is not just whitespace."""
        if not v or not v.strip():
            raise ValueError('Sentence cannot be empty or only whitespace')
        return v.strip()

    @field_validator('id')
    @classmethod
    def id_not_empty_or_whitespace(cls, v: str) -> str:
        """Validate ID is not just whitespace."""
        if not v or not v.strip():
            raise ValueError('ID cannot be empty or only whitespace')
        return v.strip()


class AnalysisRequest(BaseModel):
    """Incoming request schema for clustering analysis."""

    surveyTitle: str = Field(..., min_length=1, max_length=500, description="Survey title")
    theme: str = Field(..., min_length=1, max_length=200, description="Theme of analysis")
    baseline: List[Sentence] = Field(..., min_length=1, max_length=10000, description="Baseline sentences")
    comparison: Optional[List[Sentence]] = Field(None, max_length=10000, description="Comparison sentences")

    @field_validator('surveyTitle', 'theme')
    @classmethod
    def string_not_empty_or_whitespace(cls, v: str) -> str:
        """Validate string fields are not just whitespace."""
        if not v or not v.strip():
            raise ValueError('Field cannot be empty or only whitespace')
        return v.strip()

    def get_duplicate_ids(self):
        """Get duplicate IDs in baseline and comparison (for validation/dedup).
        
        Returns:
            tuple: (baseline_duplicates, comparison_duplicates, overlap)
        """
        baseline_ids = [s.id for s in self.baseline]
        baseline_duplicates = set([id for id in baseline_ids if baseline_ids.count(id) > 1])
        
        comparison_duplicates = set()
        overlap = set()
        
        if self.comparison:
            comparison_ids = [s.id for s in self.comparison]
            comparison_duplicates = set([id for id in comparison_ids if comparison_ids.count(id) > 1])
            overlap = set(baseline_ids) & set(comparison_ids)
        
        return baseline_duplicates, comparison_duplicates, overlap