"""Unit tests for request models."""
import pytest
from pydantic import ValidationError

from models.request import Sentence, AnalysisRequest


class TestSentence:
    """Tests for the Sentence model."""

    def test_valid_sentence(self):
        """Test creating a valid sentence."""
        sentence = Sentence(sentence="This is a test", id="test-id-1")
        assert sentence.sentence == "This is a test"
        assert sentence.id == "test-id-1"

    def test_sentence_strips_whitespace(self):
        """Test that sentence and id fields strip whitespace."""
        sentence = Sentence(sentence="  Test sentence  ", id="  test-id  ")
        assert sentence.sentence == "Test sentence"
        assert sentence.id == "test-id"

    def test_empty_sentence_raises_error(self):
        """Test that empty sentence raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Sentence(sentence="", id="test-id")
        
        # Pydantic's min_length validator runs first
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_whitespace_only_sentence_raises_error(self):
        """Test that whitespace-only sentence raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Sentence(sentence="   ", id="test-id")
        
        assert "Sentence cannot be empty" in str(exc_info.value)

    def test_empty_id_raises_error(self):
        """Test that empty id raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Sentence(sentence="Test sentence", id="")
        
        # Pydantic's min_length validator runs first
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_whitespace_only_id_raises_error(self):
        """Test that whitespace-only id raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Sentence(sentence="Test sentence", id="   ")
        
        assert "ID cannot be empty" in str(exc_info.value)

    def test_missing_sentence_field_raises_error(self):
        """Test that missing sentence field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Sentence(id="test-id")
        
        assert "sentence" in str(exc_info.value).lower()

    def test_missing_id_field_raises_error(self):
        """Test that missing id field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Sentence(sentence="Test sentence")
        
        assert "id" in str(exc_info.value).lower()

    def test_sentence_too_long_raises_error(self):
        """Test that sentence exceeding max length raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Sentence(sentence="x" * 10001, id="test-id")
        
        assert "at most 10000 characters" in str(exc_info.value).lower()

    def test_id_too_long_raises_error(self):
        """Test that id exceeding max length raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Sentence(sentence="Test", id="x" * 501)
        
        assert "at most 500 characters" in str(exc_info.value).lower()


class TestAnalysisRequest:
    """Tests for the AnalysisRequest model."""

    def test_valid_request_baseline_only(self):
        """Test creating a valid request with baseline only."""
        request = AnalysisRequest(
            surveyTitle="Test Survey",
            theme="test theme",
            baseline=[
                Sentence(sentence="First sentence", id="id-1"),
                Sentence(sentence="Second sentence", id="id-2"),
            ]
        )
        assert request.surveyTitle == "Test Survey"
        assert request.theme == "test theme"
        assert len(request.baseline) == 2
        assert request.comparison is None

    def test_valid_request_with_comparison(self):
        """Test creating a valid request with baseline and comparison."""
        request = AnalysisRequest(
            surveyTitle="Test Survey",
            theme="test theme",
            baseline=[Sentence(sentence="Baseline", id="b-1")],
            comparison=[Sentence(sentence="Comparison", id="c-1")]
        )
        assert len(request.baseline) == 1
        assert len(request.comparison) == 1

    def test_request_strips_whitespace(self):
        """Test that string fields strip whitespace."""
        request = AnalysisRequest(
            surveyTitle="  Test Survey  ",
            theme="  test theme  ",
            baseline=[Sentence(sentence="Test", id="id-1")]
        )
        assert request.surveyTitle == "Test Survey"
        assert request.theme == "test theme"

    def test_empty_survey_title_raises_error(self):
        """Test that empty survey title raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(
                surveyTitle="",
                theme="test",
                baseline=[Sentence(sentence="Test", id="id-1")]
            )
        
        # Pydantic's min_length validator runs first
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_whitespace_only_survey_title_raises_error(self):
        """Test that whitespace-only survey title raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(
                surveyTitle="   ",
                theme="test",
                baseline=[Sentence(sentence="Test", id="id-1")]
            )
        
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_empty_theme_raises_error(self):
        """Test that empty theme raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(
                surveyTitle="Test",
                theme="",
                baseline=[Sentence(sentence="Test", id="id-1")]
            )
        
        # Pydantic's min_length validator runs first
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_empty_baseline_raises_error(self):
        """Test that empty baseline raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(
                surveyTitle="Test",
                theme="test",
                baseline=[]
            )
        
        assert "at least 1" in str(exc_info.value).lower()

    def test_get_duplicate_ids_in_baseline(self):
        """Test detection of duplicate IDs in baseline."""
        request = AnalysisRequest(
            surveyTitle="Test",
            theme="test",
            baseline=[
                Sentence(sentence="First", id="duplicate-id"),
                Sentence(sentence="Second", id="duplicate-id"),
            ]
        )
        
        baseline_dups, comparison_dups, overlap = request.get_duplicate_ids()
        assert "duplicate-id" in baseline_dups
        assert len(comparison_dups) == 0
        assert len(overlap) == 0

    def test_get_duplicate_ids_in_comparison(self):
        """Test detection of duplicate IDs in comparison."""
        request = AnalysisRequest(
            surveyTitle="Test",
            theme="test",
            baseline=[Sentence(sentence="Baseline", id="b-1")],
            comparison=[
                Sentence(sentence="First", id="duplicate-id"),
                Sentence(sentence="Second", id="duplicate-id"),
            ]
        )
        
        baseline_dups, comparison_dups, overlap = request.get_duplicate_ids()
        assert len(baseline_dups) == 0
        assert "duplicate-id" in comparison_dups
        assert len(overlap) == 0

    def test_get_overlapping_ids_between_baseline_and_comparison(self):
        """Test detection of overlapping IDs between baseline and comparison."""
        request = AnalysisRequest(
            surveyTitle="Test",
            theme="test",
            baseline=[Sentence(sentence="Baseline", id="overlap-id")],
            comparison=[Sentence(sentence="Comparison", id="overlap-id")]
        )
        
        baseline_dups, comparison_dups, overlap = request.get_duplicate_ids()
        assert len(baseline_dups) == 0
        assert len(comparison_dups) == 0
        assert "overlap-id" in overlap

    def test_survey_title_too_long_raises_error(self):
        """Test that survey title exceeding max length raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(
                surveyTitle="x" * 501,
                theme="test",
                baseline=[Sentence(sentence="Test", id="id-1")]
            )
        
        assert "at most 500 characters" in str(exc_info.value).lower()

    def test_theme_too_long_raises_error(self):
        """Test that theme exceeding max length raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(
                surveyTitle="Test",
                theme="x" * 201,
                baseline=[Sentence(sentence="Test", id="id-1")]
            )
        
        assert "at most 200 characters" in str(exc_info.value).lower()

    def test_baseline_too_many_sentences_raises_error(self):
        """Test that baseline with too many sentences raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(
                surveyTitle="Test",
                theme="test",
                baseline=[Sentence(sentence=f"Sentence {i}", id=f"id-{i}") for i in range(10001)]
            )
        
        assert "at most 10000" in str(exc_info.value).lower()

    def test_missing_required_fields_raises_error(self):
        """Test that missing required fields raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(surveyTitle="Test")
        
        errors = str(exc_info.value).lower()
        assert "theme" in errors
        assert "baseline" in errors

