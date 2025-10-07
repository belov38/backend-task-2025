"""Unit tests for response models."""
import pytest
from pydantic import ValidationError

from models.response import Cluster, AnalysisResponse


class TestCluster:
    """Tests for the Cluster model."""

    def test_valid_cluster(self):
        """Test creating a valid cluster."""
        cluster = Cluster(
            title="Test Cluster",
            sentiment="positive",
            sentences=["id-1", "id-2"],
            keyInsights=["Insight 1", "Insight 2"]
        )
        assert cluster.title == "Test Cluster"
        assert cluster.sentiment == "positive"
        assert cluster.sentences == ["id-1", "id-2"]
        assert cluster.keyInsights == ["Insight 1", "Insight 2"]

    def test_cluster_with_all_sentiments(self):
        """Test creating clusters with all valid sentiment values."""
        for sentiment in ["positive", "negative", "neutral"]:
            cluster = Cluster(
                title="Test",
                sentiment=sentiment,
                sentences=["id-1"],
                keyInsights=["Insight"]
            )
            assert cluster.sentiment == sentiment

    def test_cluster_strips_whitespace(self):
        """Test that cluster fields strip whitespace."""
        cluster = Cluster(
            title="  Test Cluster  ",
            sentiment="positive",
            sentences=["  id-1  ", "  id-2  "],
            keyInsights=["  Insight 1  ", "  Insight 2  "]
        )
        assert cluster.title == "Test Cluster"
        assert cluster.sentences == ["id-1", "id-2"]
        assert cluster.keyInsights == ["Insight 1", "Insight 2"]

    def test_empty_title_raises_error(self):
        """Test that empty title raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="",
                sentiment="positive",
                sentences=["id-1"],
                keyInsights=["Insight"]
            )
        
        # Pydantic's min_length validator runs first
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_whitespace_only_title_raises_error(self):
        """Test that whitespace-only title raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="   ",
                sentiment="positive",
                sentences=["id-1"],
                keyInsights=["Insight"]
            )
        
        assert "title cannot be empty" in str(exc_info.value).lower()

    def test_invalid_sentiment_raises_error(self):
        """Test that invalid sentiment raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="Test",
                sentiment="invalid",
                sentences=["id-1"],
                keyInsights=["Insight"]
            )
        
        assert "sentiment" in str(exc_info.value).lower()

    def test_empty_sentences_list_raises_error(self):
        """Test that empty sentences list raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="Test",
                sentiment="positive",
                sentences=[],
                keyInsights=["Insight"]
            )
        
        # Pydantic's min_length validator runs first
        assert "at least 1 item" in str(exc_info.value).lower()

    def test_empty_sentence_id_raises_error(self):
        """Test that empty sentence ID raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="Test",
                sentiment="positive",
                sentences=["id-1", ""],
                keyInsights=["Insight"]
            )
        
        assert "sentence ids cannot be empty" in str(exc_info.value).lower()

    def test_whitespace_only_sentence_id_raises_error(self):
        """Test that whitespace-only sentence ID raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="Test",
                sentiment="positive",
                sentences=["id-1", "   "],
                keyInsights=["Insight"]
            )
        
        assert "sentence ids cannot be empty" in str(exc_info.value).lower()

    def test_empty_key_insights_list_raises_error(self):
        """Test that empty key insights list raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="Test",
                sentiment="positive",
                sentences=["id-1"],
                keyInsights=[]
            )
        
        # Pydantic's min_length validator runs first
        assert "at least 1 item" in str(exc_info.value).lower()

    def test_empty_key_insight_raises_error(self):
        """Test that empty key insight raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="Test",
                sentiment="positive",
                sentences=["id-1"],
                keyInsights=["Insight 1", ""]
            )
        
        assert "key insights cannot be empty" in str(exc_info.value).lower()

    def test_whitespace_only_key_insight_raises_error(self):
        """Test that whitespace-only key insight raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="Test",
                sentiment="positive",
                sentences=["id-1"],
                keyInsights=["Insight 1", "   "]
            )
        
        assert "key insights cannot be empty" in str(exc_info.value).lower()

    def test_title_too_long_raises_error(self):
        """Test that title exceeding max length raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="x" * 501,
                sentiment="positive",
                sentences=["id-1"],
                keyInsights=["Insight"]
            )
        
        assert "at most 500 characters" in str(exc_info.value).lower()

    def test_too_many_key_insights_raises_error(self):
        """Test that too many key insights raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(
                title="Test",
                sentiment="positive",
                sentences=["id-1"],
                keyInsights=[f"Insight {i}" for i in range(11)]
            )
        
        assert "at most 10" in str(exc_info.value).lower()

    def test_missing_required_fields_raises_error(self):
        """Test that missing required fields raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Cluster(title="Test")
        
        errors = str(exc_info.value).lower()
        assert "sentiment" in errors
        assert "sentences" in errors
        assert "keyinsights" in errors


class TestAnalysisResponse:
    """Tests for the AnalysisResponse model."""

    def test_valid_response(self):
        """Test creating a valid response."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Cluster 1",
                    sentiment="positive",
                    sentences=["id-1", "id-2"],
                    keyInsights=["Insight 1"]
                ),
                Cluster(
                    title="Cluster 2",
                    sentiment="negative",
                    sentences=["id-3"],
                    keyInsights=["Insight 2"]
                )
            ]
        )
        assert len(response.clusters) == 2
        assert response.clusters[0].title == "Cluster 1"
        assert response.clusters[1].title == "Cluster 2"

    def test_empty_clusters_list_is_valid(self):
        """Test that empty clusters list is valid (LLM might return no clusters)."""
        response = AnalysisResponse(clusters=[])
        assert len(response.clusters) == 0

    def test_soft_clustering_allows_duplicate_ids(self):
        """Test that soft clustering allows same sentence ID in multiple clusters."""
        # This should NOT raise an error - soft clustering is allowed
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Cluster 1",
                    sentiment="positive",
                    sentences=["id-1", "id-2"],
                    keyInsights=["Insight 1"]
                ),
                Cluster(
                    title="Cluster 2",
                    sentiment="negative",
                    sentences=["id-2", "id-3"],  # id-2 appears in both
                    keyInsights=["Insight 2"]
                )
            ]
        )
        
        # Should succeed
        assert len(response.clusters) == 2
        # id-2 appears in both clusters
        assert "id-2" in response.clusters[0].sentences
        assert "id-2" in response.clusters[1].sentences

    def test_soft_clustering_multiple_overlaps(self):
        """Test that multiple sentence IDs can appear in multiple clusters."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Cluster 1",
                    sentiment="positive",
                    sentences=["id-1", "id-2", "id-3"],
                    keyInsights=["Insight 1"]
                ),
                Cluster(
                    title="Cluster 2",
                    sentiment="negative",
                    sentences=["id-2", "id-3", "id-4"],  # Multiple overlaps
                    keyInsights=["Insight 2"]
                )
            ]
        )
        
        assert len(response.clusters) == 2
        # Both id-2 and id-3 appear in multiple clusters
        assert "id-2" in response.clusters[0].sentences
        assert "id-2" in response.clusters[1].sentences
        assert "id-3" in response.clusters[0].sentences
        assert "id-3" in response.clusters[1].sentences

    def test_response_with_single_cluster(self):
        """Test response with a single cluster."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Only Cluster",
                    sentiment="neutral",
                    sentences=["id-1"],
                    keyInsights=["Insight"]
                )
            ]
        )
        assert len(response.clusters) == 1

    def test_model_dump(self):
        """Test that model_dump() works correctly for JSON serialization."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1"],
                    keyInsights=["Insight"]
                )
            ]
        )
        dumped = response.model_dump()
        assert isinstance(dumped, dict)
        assert "clusters" in dumped
        assert len(dumped["clusters"]) == 1
        assert dumped["clusters"][0]["title"] == "Test"
        assert dumped["clusters"][0]["sentiment"] == "positive"

