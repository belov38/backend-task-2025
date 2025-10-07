"""Shared test fixtures and configuration."""
import os
import sys
import pytest

# Add src directory to Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(autouse=True)
def reset_environment_variables():
    """Reset environment variables before each test."""
    # Save original environment
    original_env = os.environ.copy()
    
    # Set default test environment variables
    os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
    os.environ.setdefault('BEDROCK_REGION', 'us-east-1')
    os.environ.setdefault('MODEL_ID', 'amazon.nova-lite-v1:0')
    os.environ.setdefault('CHUNK_SIZE', '50')
    os.environ.setdefault('MAX_WORKERS', '8')
    os.environ.setdefault('MAX_TOKENS', '1500')
    os.environ.setdefault('TEMPERATURE', '0')
    os.environ.setdefault('STOP_SEQUENCE', '}]}')
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_sentences():
    """Create a list of sample sentences for testing."""
    from models.request import Sentence
    return [
        Sentence(sentence="This is a positive review", id="pos-1"),
        Sentence(sentence="Great product, highly recommend", id="pos-2"),
        Sentence(sentence="Terrible experience, very disappointed", id="neg-1"),
        Sentence(sentence="Poor quality, not worth the money", id="neg-2"),
        Sentence(sentence="It's okay, nothing special", id="neu-1"),
    ]


@pytest.fixture
def large_sentence_set():
    """Create a large set of sentences for testing chunking."""
    from models.request import Sentence
    return [
        Sentence(sentence=f"Test sentence number {i}", id=f"id-{i}")
        for i in range(150)
    ]

