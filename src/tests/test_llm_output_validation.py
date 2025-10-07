"""Tests for LLM output validation modes (strict vs soft)."""
import pytest
from unittest.mock import Mock

from models.request import AnalysisRequest, Sentence
from models.response import AnalysisResponse, Cluster
from services.clustering import ClusteringService


@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client."""
    return Mock()


@pytest.fixture
def clustering_service(mock_bedrock_client):
    """Create a ClusteringService instance with mock client."""
    return ClusteringService(mock_bedrock_client)


@pytest.fixture
def sample_request():
    """Create a sample analysis request."""
    return AnalysisRequest(
        surveyTitle="Test Survey",
        theme="test theme",
        baseline=[
            Sentence(sentence="First", id="id-1"),
            Sentence(sentence="Second", id="id-2"),
            Sentence(sentence="Third", id="id-3"),
        ]
    )


class TestStrictLLMValidation:
    """Tests for strict LLM output validation mode."""

    def test_strict_mode_raises_error_on_invalid_ids(self, clustering_service):
        """Test that strict mode raises error when LLM returns invalid IDs."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1", "invalid-id"],  # Invalid ID
                    keyInsights=["Insight"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2", "id-3"}
        
        with pytest.raises(ValueError) as exc_info:
            clustering_service._validate_and_filter_output_sentence_ids(
                response, valid_ids, mode='strict'
            )
        
        assert "not present in input" in str(exc_info.value).lower()
        assert "invalid-id" in str(exc_info.value)

    def test_strict_mode_passes_with_valid_ids(self, clustering_service):
        """Test that strict mode passes when all IDs are valid."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1", "id-2"],
                    keyInsights=["Insight"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2", "id-3"}
        
        # Should not raise
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='strict'
        )
        assert len(result.clusters) == 1


class TestSoftLLMValidation:
    """Tests for soft LLM output validation mode."""

    def test_soft_mode_filters_invalid_ids(self, clustering_service):
        """Test that soft mode filters out invalid IDs instead of raising error."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1", "invalid-1", "id-2", "invalid-2"],
                    keyInsights=["Insight"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2", "id-3"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='soft'
        )
        
        # Invalid IDs should be filtered out
        assert result.clusters[0].sentences == ["id-1", "id-2"]
        assert "invalid-1" not in result.clusters[0].sentences
        assert "invalid-2" not in result.clusters[0].sentences

    def test_soft_mode_filters_across_multiple_clusters(self, clustering_service):
        """Test that soft mode filters invalid IDs across all clusters."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Cluster 1",
                    sentiment="positive",
                    sentences=["id-1", "bad-1"],
                    keyInsights=["I1"]
                ),
                Cluster(
                    title="Cluster 2",
                    sentiment="negative",
                    sentences=["bad-2", "id-2"],
                    keyInsights=["I2"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2", "id-3"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='soft'
        )
        
        assert result.clusters[0].sentences == ["id-1"]
        assert result.clusters[1].sentences == ["id-2"]

    def test_soft_mode_handles_all_invalid_ids_in_cluster(self, clustering_service):
        """Test soft mode when all IDs in a cluster are invalid."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["bad-1", "bad-2"],  # All invalid
                    keyInsights=["Insight"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='soft'
        )
        
        # Cluster should have empty sentences list
        assert result.clusters[0].sentences == []


class TestUnprocessedSentences:
    """Tests for unprocessed sentences tracking."""

    def test_identifies_unprocessed_sentences(self, clustering_service):
        """Test that unprocessed sentences are identified."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1", "id-2"],  # Only 2 out of 3
                    keyInsights=["Insight"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2", "id-3"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='soft'
        )
        
        # id-3 was not processed
        assert "id-3" in result.unprocessedSentences
        assert len(result.unprocessedSentences) == 1

    def test_all_sentences_processed(self, clustering_service):
        """Test that unprocessedSentences is empty when all are processed."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1", "id-2", "id-3"],
                    keyInsights=["Insight"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2", "id-3"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='soft'
        )
        
        assert result.unprocessedSentences == []

    def test_multiple_unprocessed_sentences(self, clustering_service):
        """Test tracking multiple unprocessed sentences."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1"],  # Only 1 out of 5
                    keyInsights=["Insight"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2", "id-3", "id-4", "id-5"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='soft'
        )
        
        # 4 sentences were not processed
        assert len(result.unprocessedSentences) == 4
        assert "id-2" in result.unprocessedSentences
        assert "id-3" in result.unprocessedSentences
        assert "id-4" in result.unprocessedSentences
        assert "id-5" in result.unprocessedSentences

    def test_unprocessed_with_soft_clustering(self, clustering_service):
        """Test unprocessed tracking works with soft clustering (duplicates across clusters)."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="C1",
                    sentiment="positive",
                    sentences=["id-1", "id-2"],
                    keyInsights=["I1"]
                ),
                Cluster(
                    title="C2",
                    sentiment="negative",
                    sentences=["id-2", "id-3"],  # id-2 appears in both
                    keyInsights=["I2"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2", "id-3", "id-4"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='soft'
        )
        
        # id-4 was not processed (id-2 counts even if duplicated)
        assert result.unprocessedSentences == ["id-4"]

    def test_unprocessed_sorted_alphabetically(self, clustering_service):
        """Test that unprocessed sentences are sorted."""
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
        valid_ids = {"id-1", "id-5", "id-3", "id-2"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='soft'
        )
        
        # Should be sorted
        assert result.unprocessedSentences == ["id-2", "id-3", "id-5"]


class TestMixedScenarios:
    """Tests for mixed scenarios combining invalid IDs and unprocessed sentences."""

    def test_soft_mode_filters_invalid_and_tracks_unprocessed(self, clustering_service):
        """Test that soft mode both filters invalid IDs and tracks unprocessed."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1", "invalid-id", "id-2"],
                    keyInsights=["Insight"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2", "id-3", "id-4"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='soft'
        )
        
        # Invalid ID filtered
        assert result.clusters[0].sentences == ["id-1", "id-2"]
        # Unprocessed tracked
        assert "id-3" in result.unprocessedSentences
        assert "id-4" in result.unprocessedSentences

    def test_all_invalid_ids_results_in_all_unprocessed(self, clustering_service):
        """Test when LLM returns only invalid IDs."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["bad-1", "bad-2", "bad-3"],  # All invalid
                    keyInsights=["Insight"]
                )
            ]
        )
        valid_ids = {"id-1", "id-2"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(
            response, valid_ids, mode='soft'
        )
        
        # All input IDs are unprocessed
        assert set(result.unprocessedSentences) == {"id-1", "id-2"}
        # Cluster has no valid sentences
        assert result.clusters[0].sentences == []


class TestResponseModel:
    """Tests for AnalysisResponse model with unprocessedSentences."""

    def test_response_with_unprocessed_sentences(self):
        """Test creating response with unprocessedSentences."""
        response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1"],
                    keyInsights=["Insight"]
                )
            ],
            unprocessedSentences=["id-2", "id-3"]
        )
        
        assert len(response.clusters) == 1
        assert response.unprocessedSentences == ["id-2", "id-3"]

    def test_response_without_unprocessed_sentences(self):
        """Test that unprocessedSentences defaults to empty list."""
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
        
        assert response.unprocessedSentences == []

    def test_response_model_dump_includes_unprocessed(self):
        """Test that model_dump includes unprocessedSentences."""
        response = AnalysisResponse(
            clusters=[],
            unprocessedSentences=["id-1", "id-2"]
        )
        
        dumped = response.model_dump()
        assert "unprocessedSentences" in dumped
        assert dumped["unprocessedSentences"] == ["id-1", "id-2"]

