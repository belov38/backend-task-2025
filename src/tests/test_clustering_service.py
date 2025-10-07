"""Unit tests for ClusteringService."""
import json
import pytest
from unittest.mock import Mock, patch

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
            Sentence(sentence="First test sentence", id="id-1"),
            Sentence(sentence="Second test sentence", id="id-2"),
            Sentence(sentence="Third test sentence", id="id-3"),
        ]
    )


@pytest.fixture
def sample_llm_response():
    """Create a sample LLM response."""
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


class TestClusteringServiceInit:
    """Tests for ClusteringService initialization."""

    def test_init_with_defaults(self, mock_bedrock_client):
        """Test service initialization with default environment variables."""
        service = ClusteringService(mock_bedrock_client)
        assert service.bedrock == mock_bedrock_client
        assert service.model_id == "amazon.nova-lite-v1:0"
        assert service.chunk_size == 50
        assert service.max_workers == 8
        assert service.max_tokens == 1500
        assert service.temperature == 0
        assert service.stop_sequence == "}]}"

    def test_init_with_custom_env_vars(self, mock_bedrock_client, monkeypatch):
        """Test service initialization with custom environment variables."""
        monkeypatch.setenv("MODEL_ID", "custom-model")
        monkeypatch.setenv("CHUNK_SIZE", "100")
        monkeypatch.setenv("MAX_WORKERS", "4")
        monkeypatch.setenv("MAX_TOKENS", "2000")
        monkeypatch.setenv("TEMPERATURE", "0.5")
        monkeypatch.setenv("STOP_SEQUENCE", "custom-stop")
        
        service = ClusteringService(mock_bedrock_client)
        assert service.model_id == "custom-model"
        assert service.chunk_size == 100
        assert service.max_workers == 4
        assert service.max_tokens == 2000
        assert service.temperature == 0.5
        assert service.stop_sequence == "custom-stop"


class TestChunkSentences:
    """Tests for _chunk_sentences method."""

    def test_chunk_sentences_single_chunk(self, clustering_service):
        """Test chunking when sentences fit in one chunk."""
        sentences = [Sentence(sentence=f"Sentence {i}", id=f"id-{i}") for i in range(10)]
        chunks = clustering_service._chunk_sentences(sentences)
        assert len(chunks) == 1
        assert len(chunks[0]) == 10

    def test_chunk_sentences_multiple_chunks(self, clustering_service):
        """Test chunking when sentences require multiple chunks."""
        # Default chunk size is 50
        sentences = [Sentence(sentence=f"Sentence {i}", id=f"id-{i}") for i in range(120)]
        chunks = clustering_service._chunk_sentences(sentences)
        assert len(chunks) == 3
        assert len(chunks[0]) == 50
        assert len(chunks[1]) == 50
        assert len(chunks[2]) == 20

    def test_chunk_sentences_exact_multiple(self, clustering_service):
        """Test chunking when sentence count is exact multiple of chunk size."""
        sentences = [Sentence(sentence=f"Sentence {i}", id=f"id-{i}") for i in range(100)]
        chunks = clustering_service._chunk_sentences(sentences)
        assert len(chunks) == 2
        assert len(chunks[0]) == 50
        assert len(chunks[1]) == 50


class TestNormalizeToStringList:
    """Tests for _normalize_to_string_list method."""

    def test_normalize_string_to_list(self, clustering_service):
        """Test normalizing a string to a list."""
        result = clustering_service._normalize_to_string_list("single string")
        assert result == ["single string"]

    def test_normalize_list_of_strings(self, clustering_service):
        """Test normalizing a list of strings."""
        result = clustering_service._normalize_to_string_list(["str1", "str2", "str3"])
        assert result == ["str1", "str2", "str3"]

    def test_normalize_mixed_list(self, clustering_service):
        """Test normalizing a list with mixed types filters out non-strings."""
        result = clustering_service._normalize_to_string_list(["str1", 123, "str2", None, "str3"])
        assert result == ["str1", "str2", "str3"]

    def test_normalize_non_string_non_list(self, clustering_service):
        """Test normalizing non-string, non-list returns empty list."""
        assert clustering_service._normalize_to_string_list(123) == []
        assert clustering_service._normalize_to_string_list(None) == []
        assert clustering_service._normalize_to_string_list({}) == []

    def test_normalize_empty_list(self, clustering_service):
        """Test normalizing empty list."""
        result = clustering_service._normalize_to_string_list([])
        assert result == []


class TestNormalizeSentiment:
    """Tests for _normalize_sentiment method."""

    def test_normalize_positive_sentiments(self, clustering_service):
        """Test normalizing various positive sentiment strings."""
        positive_values = ["positive", "pos", "good", "great", "excellent", "happy"]
        for value in positive_values:
            assert clustering_service._normalize_sentiment(value) == "positive"
            assert clustering_service._normalize_sentiment(value.upper()) == "positive"
            assert clustering_service._normalize_sentiment(f"  {value}  ") == "positive"

    def test_normalize_negative_sentiments(self, clustering_service):
        """Test normalizing various negative sentiment strings."""
        negative_values = ["negative", "neg", "bad", "poor", "sad", "angry"]
        for value in negative_values:
            assert clustering_service._normalize_sentiment(value) == "negative"
            assert clustering_service._normalize_sentiment(value.upper()) == "negative"

    def test_normalize_neutral_sentiments(self, clustering_service):
        """Test normalizing various neutral sentiment strings."""
        neutral_values = ["neutral", "mixed", "okay", "ok"]
        for value in neutral_values:
            assert clustering_service._normalize_sentiment(value) == "neutral"
            assert clustering_service._normalize_sentiment(value.upper()) == "neutral"

    def test_normalize_unknown_sentiment_defaults_to_neutral(self, clustering_service):
        """Test that unknown sentiment values default to neutral."""
        assert clustering_service._normalize_sentiment("unknown") == "neutral"
        assert clustering_service._normalize_sentiment("weird") == "neutral"
        assert clustering_service._normalize_sentiment("xyz") == "neutral"

    def test_normalize_non_string_sentiment_defaults_to_neutral(self, clustering_service):
        """Test that non-string sentiment values default to neutral."""
        assert clustering_service._normalize_sentiment(123) == "neutral"
        assert clustering_service._normalize_sentiment(None) == "neutral"


class TestNormalizeCluster:
    """Tests for _normalize_cluster method."""

    def test_normalize_valid_cluster(self, clustering_service):
        """Test normalizing a valid cluster dict."""
        cluster = {
            "title": "Test Cluster",
            "sentiment": "positive",
            "sentences": ["id-1", "id-2"],
            "keyInsights": ["Insight 1"]
        }
        result = clustering_service._normalize_cluster(cluster)
        assert result is not None
        assert result["title"] == "Test Cluster"
        assert result["sentiment"] == "positive"
        assert result["sentences"] == ["id-1", "id-2"]
        assert result["keyInsights"] == ["Insight 1"]

    def test_normalize_cluster_with_non_standard_sentiment(self, clustering_service):
        """Test normalizing a cluster with non-standard sentiment."""
        cluster = {
            "title": "Test",
            "sentiment": "good",
            "sentences": ["id-1"],
            "keyInsights": ["Insight"]
        }
        result = clustering_service._normalize_cluster(cluster)
        assert result["sentiment"] == "positive"

    def test_normalize_cluster_with_string_sentences(self, clustering_service):
        """Test normalizing cluster when sentences is a string."""
        cluster = {
            "title": "Test",
            "sentiment": "positive",
            "sentences": "id-1",
            "keyInsights": ["Insight"]
        }
        result = clustering_service._normalize_cluster(cluster)
        assert result["sentences"] == ["id-1"]

    def test_normalize_cluster_with_string_insights(self, clustering_service):
        """Test normalizing cluster when keyInsights is a string."""
        cluster = {
            "title": "Test",
            "sentiment": "positive",
            "sentences": ["id-1"],
            "keyInsights": "Single insight"
        }
        result = clustering_service._normalize_cluster(cluster)
        assert result["keyInsights"] == ["Single insight"]

    def test_normalize_non_dict_returns_none(self, clustering_service):
        """Test that non-dict input returns None."""
        assert clustering_service._normalize_cluster("not a dict") is None
        assert clustering_service._normalize_cluster(123) is None
        assert clustering_service._normalize_cluster(None) is None

    def test_normalize_cluster_missing_title_returns_none(self, clustering_service):
        """Test that cluster missing title returns None."""
        cluster = {
            "sentiment": "positive",
            "sentences": ["id-1"],
            "keyInsights": ["Insight"]
        }
        result = clustering_service._normalize_cluster(cluster)
        assert result is None

    def test_normalize_cluster_non_string_title_returns_none(self, clustering_service):
        """Test that cluster with non-string title returns None."""
        cluster = {
            "title": 123,
            "sentiment": "positive",
            "sentences": ["id-1"],
            "keyInsights": ["Insight"]
        }
        result = clustering_service._normalize_cluster(cluster)
        assert result is None

    def test_normalize_cluster_missing_sentiment_returns_none(self, clustering_service):
        """Test that cluster missing sentiment returns None."""
        cluster = {
            "title": "Test",
            "sentences": ["id-1"],
            "keyInsights": ["Insight"]
        }
        result = clustering_service._normalize_cluster(cluster)
        assert result is None


class TestParseJson:
    """Tests for _parse_json method."""

    def test_parse_valid_json(self, clustering_service):
        """Test parsing valid JSON."""
        json_text = json.dumps({
            "clusters": [
                {
                    "title": "Test",
                    "sentiment": "positive",
                    "sentences": ["id-1"],
                    "keyInsights": ["Insight"]
                }
            ]
        })
        result = clustering_service._parse_json(json_text)
        assert "clusters" in result
        assert len(result["clusters"]) == 1

    def test_parse_json_with_markdown_code_fence(self, clustering_service):
        """Test parsing JSON wrapped in markdown code fence."""
        json_text = '```json\n{"clusters": []}\n```'
        result = clustering_service._parse_json(json_text)
        assert result == {"clusters": []}

    def test_parse_json_with_extra_text(self, clustering_service):
        """Test parsing JSON with extra text before/after."""
        json_text = 'Here is the result:\n{"clusters": []}\nEnd of result'
        result = clustering_service._parse_json(json_text)
        assert result == {"clusters": []}

    def test_parse_invalid_json_returns_empty_clusters(self, clustering_service):
        """Test that invalid JSON returns empty clusters list."""
        result = clustering_service._parse_json("not valid json")
        assert result == {"clusters": []}

    def test_parse_json_missing_clusters_key_returns_empty(self, clustering_service):
        """Test that JSON missing clusters key returns empty list."""
        json_text = json.dumps({"other": "data"})
        result = clustering_service._parse_json(json_text)
        assert result == {"clusters": []}

    def test_parse_json_clusters_not_list_returns_empty(self, clustering_service):
        """Test that clusters as non-list returns empty list."""
        json_text = json.dumps({"clusters": "not a list"})
        result = clustering_service._parse_json(json_text)
        assert result == {"clusters": []}

    def test_parse_json_filters_invalid_clusters(self, clustering_service):
        """Test that invalid clusters are filtered out."""
        json_text = json.dumps({
            "clusters": [
                {"title": "Valid", "sentiment": "positive", "sentences": ["id-1"], "keyInsights": ["I"]},
                {"invalid": "cluster"},
                {"title": "Valid2", "sentiment": "negative", "sentences": ["id-2"], "keyInsights": ["I2"]}
            ]
        })
        result = clustering_service._parse_json(json_text)
        assert len(result["clusters"]) == 2


class TestMergeClusters:
    """Tests for _merge_clusters method."""

    def test_merge_no_overlap(self, clustering_service):
        """Test merging clusters with no title overlap."""
        chunk_results = [
            {
                "clusters": [
                    {"title": "Cluster A", "sentiment": "positive", "sentences": ["id-1"], "keyInsights": ["I1"]}
                ]
            },
            {
                "clusters": [
                    {"title": "Cluster B", "sentiment": "negative", "sentences": ["id-2"], "keyInsights": ["I2"]}
                ]
            }
        ]
        merged = clustering_service._merge_clusters(chunk_results)
        assert len(merged) == 2

    def test_merge_with_same_title(self, clustering_service):
        """Test merging clusters with same title."""
        chunk_results = [
            {
                "clusters": [
                    {"title": "Same Title", "sentiment": "positive", "sentences": ["id-1"], "keyInsights": ["I1"]}
                ]
            },
            {
                "clusters": [
                    {"title": "Same Title", "sentiment": "positive", "sentences": ["id-2"], "keyInsights": ["I2"]}
                ]
            }
        ]
        merged = clustering_service._merge_clusters(chunk_results)
        assert len(merged) == 1
        assert len(merged[0].sentences) == 2
        assert len(merged[0].keyInsights) == 2

    def test_merge_title_case_insensitive(self, clustering_service):
        """Test that merge is case-insensitive for titles."""
        chunk_results = [
            {
                "clusters": [
                    {"title": "Test Cluster", "sentiment": "positive", "sentences": ["id-1"], "keyInsights": ["I1"]}
                ]
            },
            {
                "clusters": [
                    {"title": "test cluster", "sentiment": "neutral", "sentences": ["id-2"], "keyInsights": ["I2"]}
                ]
            }
        ]
        merged = clustering_service._merge_clusters(chunk_results)
        assert len(merged) == 1

    def test_merge_negative_sentiment_takes_precedence(self, clustering_service):
        """Test that negative sentiment takes precedence when merging."""
        chunk_results = [
            {
                "clusters": [
                    {"title": "Test", "sentiment": "positive", "sentences": ["id-1"], "keyInsights": ["I1"]}
                ]
            },
            {
                "clusters": [
                    {"title": "Test", "sentiment": "negative", "sentences": ["id-2"], "keyInsights": ["I2"]}
                ]
            }
        ]
        merged = clustering_service._merge_clusters(chunk_results)
        assert merged[0].sentiment == "negative"

    def test_merge_deduplicates_insights(self, clustering_service):
        """Test that duplicate insights are removed when merging."""
        chunk_results = [
            {
                "clusters": [
                    {"title": "Test", "sentiment": "positive", "sentences": ["id-1"], "keyInsights": ["Common", "Unique1"]}
                ]
            },
            {
                "clusters": [
                    {"title": "Test", "sentiment": "positive", "sentences": ["id-2"], "keyInsights": ["Common", "Unique2"]}
                ]
            }
        ]
        merged = clustering_service._merge_clusters(chunk_results)
        assert len(merged[0].keyInsights) == 3
        assert "Common" in merged[0].keyInsights
        assert "Unique1" in merged[0].keyInsights
        assert "Unique2" in merged[0].keyInsights

    def test_merge_empty_results(self, clustering_service):
        """Test merging empty results."""
        merged = clustering_service._merge_clusters([])
        assert merged == []

    def test_merge_results_with_empty_clusters(self, clustering_service):
        """Test merging results where some have empty clusters."""
        chunk_results = [
            {"clusters": []},
            {
                "clusters": [
                    {"title": "Test", "sentiment": "positive", "sentences": ["id-1"], "keyInsights": ["I1"]}
                ]
            }
        ]
        merged = clustering_service._merge_clusters(chunk_results)
        assert len(merged) == 1


class TestValidateAndFilterOutputSentenceIds:
    """Tests for _validate_and_filter_output_sentence_ids method."""

    def test_validate_all_ids_valid_strict_mode(self, clustering_service):
        """Test validation passes when all IDs are valid in strict mode."""
        response = AnalysisResponse(
            clusters=[
                Cluster(title="C1", sentiment="positive", sentences=["id-1", "id-2"], keyInsights=["I1"])
            ]
        )
        valid_ids = {"id-1", "id-2", "id-3"}
        # Should not raise
        result = clustering_service._validate_and_filter_output_sentence_ids(response, valid_ids, mode='strict')
        assert len(result.clusters) == 1
        assert result.unprocessedSentences == ["id-3"]

    def test_validate_invalid_id_raises_error_strict_mode(self, clustering_service):
        """Test validation raises error when ID is invalid in strict mode."""
        response = AnalysisResponse(
            clusters=[
                Cluster(title="C1", sentiment="positive", sentences=["id-1", "invalid-id"], keyInsights=["I1"])
            ]
        )
        valid_ids = {"id-1", "id-2"}
        
        with pytest.raises(ValueError) as exc_info:
            clustering_service._validate_and_filter_output_sentence_ids(response, valid_ids, mode='strict')
        
        assert "not present in input" in str(exc_info.value).lower()
        assert "invalid-id" in str(exc_info.value)

    def test_validate_filters_invalid_ids_soft_mode(self, clustering_service):
        """Test validation filters invalid IDs in soft mode."""
        response = AnalysisResponse(
            clusters=[
                Cluster(title="C1", sentiment="positive", sentences=["id-1", "bad-1"], keyInsights=["I1"]),
                Cluster(title="C2", sentiment="negative", sentences=["bad-2", "id-2"], keyInsights=["I2"])
            ]
        )
        valid_ids = {"id-1", "id-2"}
        
        result = clustering_service._validate_and_filter_output_sentence_ids(response, valid_ids, mode='soft')
        
        # Invalid IDs should be filtered
        assert result.clusters[0].sentences == ["id-1"]
        assert result.clusters[1].sentences == ["id-2"]
        assert result.unprocessedSentences == []

    def test_validate_empty_clusters(self, clustering_service):
        """Test validation with empty clusters."""
        response = AnalysisResponse(clusters=[])
        valid_ids = {"id-1", "id-2"}
        # Should not raise
        result = clustering_service._validate_and_filter_output_sentence_ids(response, valid_ids, mode='strict')
        # All input IDs are unprocessed
        assert set(result.unprocessedSentences) == valid_ids


class TestAnalyze:
    """Tests for the analyze method."""

    def test_analyze_small_dataset_single_call(self, clustering_service, sample_request, sample_llm_response, mock_bedrock_client):
        """Test analyze with small dataset uses single call."""
        mock_bedrock_client.converse.return_value = sample_llm_response
        
        result = clustering_service.analyze(sample_request)
        
        assert isinstance(result, AnalysisResponse)
        assert len(result.clusters) == 1
        assert mock_bedrock_client.converse.call_count == 1

    def test_analyze_validates_output_ids_strict_mode(self, clustering_service, sample_request, mock_bedrock_client):
        """Test that analyze validates output sentence IDs in strict mode."""
        # Return response with invalid ID
        invalid_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": json.dumps({
                                "clusters": [
                                    {
                                        "title": "Test",
                                        "sentiment": "positive",
                                        "sentences": ["invalid-id"],
                                        "keyInsights": ["Insight"]
                                    }
                                ]
                            })
                        }
                    ]
                }
            }
        }
        mock_bedrock_client.converse.return_value = invalid_response
        
        with patch.dict('os.environ', {'LLM_OUTPUT_VALIDATION': 'strict'}):
            with pytest.raises(ValueError) as exc_info:
                clustering_service.analyze(sample_request)
            
            assert "not present in input" in str(exc_info.value).lower()
    
    def test_analyze_filters_invalid_ids_soft_mode(self, clustering_service, sample_request, mock_bedrock_client, sample_llm_response):
        """Test that analyze filters invalid IDs in soft mode."""
        # Return response with mixed valid and invalid IDs
        mixed_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": json.dumps({
                                "clusters": [
                                    {
                                        "title": "Test",
                                        "sentiment": "positive",
                                        "sentences": [sample_request.baseline[0].id, "invalid-id"],
                                        "keyInsights": ["Insight"]
                                    }
                                ]
                            })
                        }
                    ]
                }
            }
        }
        mock_bedrock_client.converse.return_value = mixed_response
        
        with patch.dict('os.environ', {'LLM_OUTPUT_VALIDATION': 'soft'}):
            result = clustering_service.analyze(sample_request)
            
            # Should succeed and filter invalid ID
            assert len(result.clusters) == 1
            assert sample_request.baseline[0].id in result.clusters[0].sentences
            assert "invalid-id" not in result.clusters[0].sentences

