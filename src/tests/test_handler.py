"""Integration tests for the Lambda handler."""
import json
import pytest
from unittest.mock import Mock, patch

from handler import analyze, lambda_handler
from models.response import AnalysisResponse, Cluster


@pytest.fixture
def mock_bedrock_response():
    """Create a mock Bedrock response."""
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "text": json.dumps({
                            "clusters": [
                                {
                                    "title": "Test Cluster",
                                    "sentiment": "positive",
                                    "sentences": ["id-1", "id-2"],
                                    "keyInsights": ["Insight 1", "Insight 2"]
                                }
                            ]
                        })
                    }
                ]
            }
        }
    }


@pytest.fixture
def sample_api_gateway_event():
    """Create a sample API Gateway event."""
    return {
        "resource": "/analyze",
        "path": "/analyze",
        "httpMethod": "POST",
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "surveyTitle": "Test Survey",
            "theme": "test theme",
            "baseline": [
                {"sentence": "First test sentence", "id": "id-1"},
                {"sentence": "Second test sentence", "id": "id-2"},
            ]
        }),
        "isBase64Encoded": False
    }


@pytest.fixture
def invalid_api_gateway_event():
    """Create an invalid API Gateway event (missing required fields)."""
    return {
        "resource": "/analyze",
        "path": "/analyze",
        "httpMethod": "POST",
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "surveyTitle": "Test Survey",
            # Missing theme and baseline
        }),
        "isBase64Encoded": False
    }


class TestAnalyzeEndpoint:
    """Tests for the /analyze endpoint."""

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_analyze_success(self, mock_app, mock_clustering_service):
        """Test successful analysis."""
        # Setup mock
        mock_request_data = {
            "surveyTitle": "Test Survey",
            "theme": "test",
            "baseline": [
                {"sentence": "Test", "id": "id-1"}
            ]
        }
        mock_app.current_event.json_body = mock_request_data
        
        mock_response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1"],
                    keyInsights=["Insight"]
                )
            ]
        )
        mock_clustering_service.analyze.return_value = mock_response
        
        # Call the endpoint
        result = analyze()
        
        # Assertions
        assert result["statusCode"] == 200
        assert "body" in result
        assert result["body"]["clusters"]
        mock_clustering_service.analyze.assert_called_once()

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_analyze_validation_error(self, mock_app, mock_clustering_service):
        """Test analysis with validation error."""
        from pydantic import ValidationError
        
        # Setup mock to raise validation error
        mock_app.current_event.json_body = {"invalid": "data"}
        
        # Call the endpoint directly - it will catch ValidationError
        # We need to simulate what happens when AnalysisRequest(**data) fails
        with patch('handler.AnalysisRequest') as mock_request:
            mock_request.side_effect = ValidationError.from_exception_data(
                "ValidationError",
                [{"type": "missing", "loc": ("theme",), "msg": "Field required", "input": {}}]
            )
            
            result = analyze()
            
            # Assertions
            assert result["statusCode"] == 400
            assert "error" in result["body"]
            assert result["body"]["error"] == "Invalid request"

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_analyze_internal_error(self, mock_app, mock_clustering_service):
        """Test analysis with internal error."""
        # Setup mock
        mock_app.current_event.json_body = {
            "surveyTitle": "Test",
            "theme": "test",
            "baseline": [{"sentence": "Test", "id": "id-1"}]
        }
        
        # Make clustering service raise an exception
        mock_clustering_service.analyze.side_effect = Exception("Internal error")
        
        # Call the endpoint
        result = analyze()
        
        # Assertions
        assert result["statusCode"] == 500
        assert "error" in result["body"]
        assert result["body"]["error"] == "Internal error"


class TestLambdaHandler:
    """Tests for the lambda_handler function."""

    @patch('handler.app')
    def test_lambda_handler_delegates_to_resolver(self, mock_app):
        """Test that lambda_handler delegates to the API Gateway resolver."""
        mock_event = {"test": "event"}
        mock_context = Mock()
        mock_app.resolve.return_value = {"statusCode": 200}
        
        result = lambda_handler(mock_event, mock_context)
        
        mock_app.resolve.assert_called_once_with(mock_event, mock_context)
        assert result == {"statusCode": 200}


class TestInputValidation:
    """Tests for input validation at the handler level."""

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_empty_survey_title_rejected(self, mock_app, mock_clustering_service):
        """Test that empty survey title is rejected."""
        # Use actual validation - it will fail naturally
        mock_app.current_event.json_body = {
            "surveyTitle": "",
            "theme": "test",
            "baseline": [{"sentence": "Test", "id": "id-1"}]
        }
        
        # This will trigger real validation error
        result = analyze()
        assert result["statusCode"] == 400
        assert "error" in result["body"]

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_duplicate_ids_rejected_in_strict_mode(self, mock_app, mock_clustering_service):
        """Test that duplicate sentence IDs are rejected in strict mode."""
        mock_app.current_event.json_body = {
            "surveyTitle": "Test",
            "theme": "test",
            "baseline": [
                {"sentence": "First", "id": "duplicate"},
                {"sentence": "Second", "id": "duplicate"}
            ]
        }
        
        # Ensure strict mode
        with patch.dict('os.environ', {'VALIDATION_MODE': 'strict'}):
            result = analyze()
        
        assert result["statusCode"] == 400
        assert "error" in result["body"]

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_empty_baseline_rejected(self, mock_app, mock_clustering_service):
        """Test that empty baseline is rejected."""
        # Use actual validation - it will fail naturally
        mock_app.current_event.json_body = {
            "surveyTitle": "Test",
            "theme": "test",
            "baseline": []
        }
        
        # This will trigger real validation error
        result = analyze()
        assert result["statusCode"] == 400
        assert "error" in result["body"]


class TestOutputValidation:
    """Tests for output validation at the handler level."""

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_handler_returns_valid_response_structure(self, mock_app, mock_clustering_service):
        """Test that handler returns valid response structure."""
        mock_app.current_event.json_body = {
            "surveyTitle": "Test",
            "theme": "test",
            "baseline": [{"sentence": "Test", "id": "id-1"}]
        }
        
        mock_response = AnalysisResponse(
            clusters=[
                Cluster(
                    title="Test",
                    sentiment="positive",
                    sentences=["id-1"],
                    keyInsights=["Insight"]
                )
            ]
        )
        mock_clustering_service.analyze.return_value = mock_response
        
        result = analyze()
        
        assert result["statusCode"] == 200
        assert "body" in result
        assert "clusters" in result["body"]
        assert isinstance(result["body"]["clusters"], list)
        
        cluster = result["body"]["clusters"][0]
        assert "title" in cluster
        assert "sentiment" in cluster
        assert "sentences" in cluster
        assert "keyInsights" in cluster

    @patch('handler.clustering_service')
    @patch('handler.app')
    def test_llm_output_validation_failure_returns_500(self, mock_app, mock_clustering_service):
        """Test that LLM output validation failure returns 500."""
        mock_app.current_event.json_body = {
            "surveyTitle": "Test",
            "theme": "test",
            "baseline": [{"sentence": "Test", "id": "id-1"}]
        }
        
        # Make clustering service raise ValueError (from output validation)
        mock_clustering_service.analyze.side_effect = ValueError("Invalid sentence IDs in LLM output")
        
        result = analyze()
        
        assert result["statusCode"] == 500
        assert "error" in result["body"]

