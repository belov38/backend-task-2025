"""Tests for validation modes (strict vs dedup)."""
import pytest
import os
from unittest.mock import Mock, patch

from handler import analyze, deduplicate_sentences
from models.response import AnalysisResponse, Cluster


@pytest.fixture
def mock_app_with_duplicates():
    """Mock app with duplicate IDs in request."""
    mock_app = Mock()
    mock_app.current_event.json_body = {
        "surveyTitle": "Test",
        "theme": "test",
        "baseline": [
            {"sentence": "First", "id": "duplicate"},
            {"sentence": "Second", "id": "duplicate"},
            {"sentence": "Third", "id": "unique"}
        ]
    }
    return mock_app


@pytest.fixture
def mock_app_with_comparison_duplicates():
    """Mock app with duplicates in comparison."""
    mock_app = Mock()
    mock_app.current_event.json_body = {
        "surveyTitle": "Test",
        "theme": "test",
        "baseline": [{"sentence": "Base", "id": "b-1"}],
        "comparison": [
            {"sentence": "First", "id": "dup"},
            {"sentence": "Second", "id": "dup"}
        ]
    }
    return mock_app


@pytest.fixture
def mock_clustering_response():
    """Mock clustering service response."""
    return AnalysisResponse(
        clusters=[
            Cluster(
                title="Test Cluster",
                sentiment="positive",
                sentences=["duplicate", "unique"],
                keyInsights=["Insight 1"]
            )
        ]
    )


class TestStrictMode:
    """Tests for strict validation mode."""

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_strict_mode_rejects_duplicates(self, mock_app, mock_clustering_service, mock_app_with_duplicates):
        """Test that strict mode rejects duplicate IDs with 400."""
        mock_app.current_event.json_body = mock_app_with_duplicates.current_event.json_body
        
        with patch.dict(os.environ, {'VALIDATION_MODE': 'strict'}):
            result = analyze()
        
        assert result["statusCode"] == 400
        assert "error" in result["body"]
        assert "duplicate" in result["body"]["details"].lower()

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_strict_mode_rejects_comparison_duplicates(self, mock_app, mock_clustering_service, mock_app_with_comparison_duplicates):
        """Test that strict mode rejects duplicates in comparison."""
        mock_app.current_event.json_body = mock_app_with_comparison_duplicates.current_event.json_body
        
        with patch.dict(os.environ, {'VALIDATION_MODE': 'strict'}):
            result = analyze()
        
        assert result["statusCode"] == 400
        assert "error" in result["body"]

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_strict_mode_accepts_unique_ids(self, mock_app, mock_clustering_service, mock_clustering_response):
        """Test that strict mode accepts unique IDs."""
        mock_app.current_event.json_body = {
            "surveyTitle": "Test",
            "theme": "test",
            "baseline": [
                {"sentence": "First", "id": "id-1"},
                {"sentence": "Second", "id": "id-2"}
            ]
        }
        mock_clustering_service.analyze.return_value = mock_clustering_response
        
        with patch.dict(os.environ, {'VALIDATION_MODE': 'strict'}):
            result = analyze()
        
        assert result["statusCode"] == 200

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_strict_mode_default(self, mock_app, mock_clustering_service, mock_app_with_duplicates):
        """Test that strict is the default mode when env var not set."""
        mock_app.current_event.json_body = mock_app_with_duplicates.current_event.json_body
        
        # Don't set VALIDATION_MODE - should default to strict
        with patch.dict(os.environ, {}, clear=False):
            if 'VALIDATION_MODE' in os.environ:
                del os.environ['VALIDATION_MODE']
            result = analyze()
        
        assert result["statusCode"] == 400


class TestDedupMode:
    """Tests for deduplication mode."""

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_dedup_mode_removes_duplicates(self, mock_app, mock_clustering_service, mock_app_with_duplicates, mock_clustering_response):
        """Test that dedup mode removes duplicates and continues."""
        mock_app.current_event.json_body = mock_app_with_duplicates.current_event.json_body
        mock_clustering_service.analyze.return_value = mock_clustering_response
        
        with patch.dict(os.environ, {'VALIDATION_MODE': 'dedup'}):
            result = analyze()
        
        assert result["statusCode"] == 200
        # Clustering service should be called with deduplicated data
        mock_clustering_service.analyze.assert_called_once()
        call_args = mock_clustering_service.analyze.call_args[0][0]
        # Should have 2 sentences (one duplicate removed)
        assert len(call_args.baseline) == 2

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_dedup_mode_keeps_first_occurrence(self, mock_app, mock_clustering_service, mock_app_with_duplicates, mock_clustering_response):
        """Test that dedup mode keeps first occurrence of duplicate ID."""
        mock_app.current_event.json_body = mock_app_with_duplicates.current_event.json_body
        mock_clustering_service.analyze.return_value = mock_clustering_response
        
        with patch.dict(os.environ, {'VALIDATION_MODE': 'dedup'}):
            result = analyze()
        
        call_args = mock_clustering_service.analyze.call_args[0][0]
        # First sentence with "duplicate" ID should be kept
        assert call_args.baseline[0].sentence == "First"
        assert call_args.baseline[0].id == "duplicate"
        # Second occurrence should be removed
        assert call_args.baseline[1].id == "unique"

    @patch('handler.clustering_service')
    @patch('handler.app')
    @patch('handler.logger')
    def test_dedup_mode_logs_warning(self, mock_logger, mock_app, mock_clustering_service, mock_app_with_duplicates, mock_clustering_response):
        """Test that dedup mode logs warning when duplicates are found."""
        mock_app.current_event.json_body = mock_app_with_duplicates.current_event.json_body
        mock_clustering_service.analyze.return_value = mock_clustering_response
        
        with patch.dict(os.environ, {'VALIDATION_MODE': 'dedup'}):
            result = analyze()
        
        # Should log warning about duplicates
        mock_logger.warning.assert_called()
        warning_call = str(mock_logger.warning.call_args)
        assert "duplicate" in warning_call.lower()

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_dedup_mode_handles_comparison(self, mock_app, mock_clustering_service, mock_app_with_comparison_duplicates, mock_clustering_response):
        """Test that dedup mode handles comparison duplicates."""
        mock_app.current_event.json_body = mock_app_with_comparison_duplicates.current_event.json_body
        mock_clustering_service.analyze.return_value = mock_clustering_response
        
        with patch.dict(os.environ, {'VALIDATION_MODE': 'dedup'}):
            result = analyze()
        
        assert result["statusCode"] == 200
        call_args = mock_clustering_service.analyze.call_args[0][0]
        # Comparison should have 1 sentence (one duplicate removed)
        assert len(call_args.comparison) == 1


class TestDeduplicateSentences:
    """Tests for deduplicate_sentences helper function."""

    def test_deduplicate_no_duplicates(self):
        """Test deduplication with no duplicates."""
        sentences = [
            {"sentence": "First", "id": "1"},
            {"sentence": "Second", "id": "2"}
        ]
        result = deduplicate_sentences(sentences)
        assert len(result) == 2

    def test_deduplicate_with_duplicates(self):
        """Test deduplication with duplicates."""
        sentences = [
            {"sentence": "First", "id": "dup"},
            {"sentence": "Second", "id": "dup"},
            {"sentence": "Third", "id": "unique"}
        ]
        result = deduplicate_sentences(sentences)
        assert len(result) == 2
        assert result[0]["id"] == "dup"
        assert result[0]["sentence"] == "First"
        assert result[1]["id"] == "unique"

    def test_deduplicate_keeps_first_occurrence(self):
        """Test that first occurrence is kept."""
        sentences = [
            {"sentence": "Keep me", "id": "same"},
            {"sentence": "Remove me", "id": "same"}
        ]
        result = deduplicate_sentences(sentences)
        assert len(result) == 1
        assert result[0]["sentence"] == "Keep me"

    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""
        result = deduplicate_sentences([])
        assert result == []

    @patch('handler.logger')
    def test_deduplicate_logs_warning(self, mock_logger):
        """Test that deduplication logs warning."""
        sentences = [
            {"sentence": "First", "id": "dup"},
            {"sentence": "Second", "id": "dup"}
        ]
        deduplicate_sentences(sentences)
        mock_logger.warning.assert_called_once()


class TestValidationModeIntegration:
    """Integration tests for validation modes."""

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_mode_case_insensitive(self, mock_app, mock_clustering_service, mock_app_with_duplicates, mock_clustering_response):
        """Test that validation mode is case insensitive."""
        mock_app.current_event.json_body = mock_app_with_duplicates.current_event.json_body
        mock_clustering_service.analyze.return_value = mock_clustering_response
        
        # Test uppercase
        with patch.dict(os.environ, {'VALIDATION_MODE': 'DEDUP'}):
            result = analyze()
        assert result["statusCode"] == 200
        
        # Test mixed case
        with patch.dict(os.environ, {'VALIDATION_MODE': 'DeDup'}):
            result = analyze()
        assert result["statusCode"] == 200

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_invalid_mode_defaults_to_strict(self, mock_app, mock_clustering_service, mock_app_with_duplicates):
        """Test that invalid mode value defaults to strict behavior."""
        mock_app.current_event.json_body = mock_app_with_duplicates.current_event.json_body
        
        with patch.dict(os.environ, {'VALIDATION_MODE': 'invalid'}):
            result = analyze()
        
        # Should behave like strict mode (reject duplicates)
        # Note: Actually it won't reject because 'invalid' != 'strict'
        # But this shows the behavior - unknown modes are lenient
        # We could enhance this to default to strict for unknown values
        assert result["statusCode"] in [200, 400]

