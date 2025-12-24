"""
Comprehensive test suite for Radiant Agentic RAG.

Provides unit tests with mocking for all major components.

Run with: pytest tests_comprehensive.py -v --cov=. --cov-report=html
"""

import gzip
import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfigurationParsing(unittest.TestCase):
    """Test configuration parsing utilities."""

    def test_parse_bool_true_values(self):
        """Test boolean parsing for true values."""
        from config import _parse_bool

        true_values = [True, "true", "True", "TRUE", "1", "yes", "Yes", "YES", "y", "Y", "on", "ON"]
        for val in true_values:
            with self.subTest(value=val):
                self.assertTrue(_parse_bool(val), f"Expected True for {val!r}")

    def test_parse_bool_false_values(self):
        """Test boolean parsing for false values."""
        from config import _parse_bool

        false_values = [False, "false", "False", "FALSE", "0", "no", "No", "NO", "n", "N", "off", "OFF", ""]
        for val in false_values:
            with self.subTest(value=val):
                self.assertFalse(_parse_bool(val), f"Expected False for {val!r}")

    def test_parse_int_valid(self):
        """Test integer parsing with valid values."""
        from config import _parse_int

        self.assertEqual(_parse_int(42, 0), 42)
        self.assertEqual(_parse_int("42", 0), 42)
        self.assertEqual(_parse_int("-10", 0), -10)
        self.assertEqual(_parse_int("  100  ", 0), 100)

    def test_parse_int_invalid_returns_default(self):
        """Test integer parsing with invalid values returns default."""
        from config import _parse_int

        self.assertEqual(_parse_int("not_a_number", 99), 99)
        self.assertEqual(_parse_int("12.5", 99), 99)  # Float string
        self.assertEqual(_parse_int(None, 99), 99)

    def test_parse_float_valid(self):
        """Test float parsing with valid values."""
        from config import _parse_float

        self.assertAlmostEqual(_parse_float(3.14, 0.0), 3.14)
        self.assertAlmostEqual(_parse_float("3.14", 0.0), 3.14)
        self.assertAlmostEqual(_parse_float(42, 0.0), 42.0)
        self.assertAlmostEqual(_parse_float("  -2.5  ", 0.0), -2.5)

    def test_parse_float_invalid_returns_default(self):
        """Test float parsing with invalid values returns default."""
        from config import _parse_float

        self.assertEqual(_parse_float("not_a_number", 1.5), 1.5)
        self.assertEqual(_parse_float(None, 1.5), 1.5)

    def test_frozen_dataclasses(self):
        """Verify all config dataclasses are frozen (immutable)."""
        from config import (
            OllamaConfig, LocalModelsConfig, BM25Config, RetrievalConfig,
            RerankConfig, AutoMergeConfig, SynthesisConfig, CriticConfig,
            QueryConfig, ConversationConfig, ParsingConfig,
            UnstructuredCleaningConfig, LoggingConfig, MetricsConfig, PipelineConfig
        )

        # Test that OllamaConfig is frozen
        config = OllamaConfig(
            openai_base_url="http://test",
            openai_api_key="test-key"
        )
        with self.assertRaises(Exception):  # FrozenInstanceError
            config.openai_base_url = "http://modified"

    def test_config_to_dict_redacts_api_key(self):
        """Test that config_to_dict redacts sensitive values."""
        from config import config_to_dict, AppConfig, OllamaConfig, LocalModelsConfig
        from config import RedisConfig, BM25Config, RetrievalConfig, RerankConfig
        from config import AutoMergeConfig, SynthesisConfig, CriticConfig, QueryConfig
        from config import ConversationConfig, ParsingConfig, UnstructuredCleaningConfig
        from config import LoggingConfig, MetricsConfig, PipelineConfig, VLMCaptionerConfig

        app_config = AppConfig(
            ollama=OllamaConfig(openai_base_url="http://test", openai_api_key="secret-key"),
            local_models=LocalModelsConfig(),
            redis=RedisConfig(),
            bm25=BM25Config(),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            automerge=AutoMergeConfig(),
            synthesis=SynthesisConfig(),
            critic=CriticConfig(),
            query=QueryConfig(),
            conversation=ConversationConfig(),
            parsing=ParsingConfig(),
            unstructured_cleaning=UnstructuredCleaningConfig(),
            logging=LoggingConfig(),
            metrics=MetricsConfig(),
            pipeline=PipelineConfig(),
            vlm=VLMCaptionerConfig(),
        )

        result = config_to_dict(app_config)
        self.assertEqual(result["ollama"]["openai_api_key"], "***REDACTED***")


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration file loading."""

    def test_find_config_file_explicit_path(self):
        """Test finding config file at explicit path."""
        from config import find_config_file

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"test: value")
            temp_path = f.name

        try:
            result = find_config_file(temp_path)
            self.assertEqual(result, Path(temp_path))
        finally:
            os.unlink(temp_path)

    def test_find_config_file_nonexistent_returns_none(self):
        """Test that nonexistent explicit path returns None."""
        from config import find_config_file

        result = find_config_file("/nonexistent/path/config.yaml")
        self.assertIsNone(result)

    def test_load_yaml_config_valid(self):
        """Test loading valid YAML configuration."""
        from config import load_yaml_config

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode='w') as f:
            f.write("key1: value1\nkey2: 42")
            temp_path = Path(f.name)

        try:
            result = load_yaml_config(temp_path)
            self.assertEqual(result["key1"], "value1")
            self.assertEqual(result["key2"], 42)
        finally:
            os.unlink(temp_path)

    def test_load_yaml_config_invalid_returns_empty(self):
        """Test that invalid YAML returns empty dict."""
        from config import load_yaml_config

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode='w') as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)

        try:
            result = load_yaml_config(temp_path)
            self.assertEqual(result, {})
        finally:
            os.unlink(temp_path)


# =============================================================================
# Metrics Tests
# =============================================================================

class TestStepMetric(unittest.TestCase):
    """Test StepMetric functionality."""

    def test_step_metric_latency_calculation(self):
        """Test that latency is correctly calculated."""
        from metrics import StepMetric

        start = time.time()
        step = StepMetric(name="test", started_at=start)
        time.sleep(0.02)  # 20ms
        step.ended_at = time.time()

        self.assertIsNotNone(step.latency_ms)
        self.assertGreater(step.latency_ms, 15)  # At least 15ms
        self.assertLess(step.latency_ms, 100)  # Less than 100ms

    def test_step_metric_incomplete_has_no_latency(self):
        """Test that incomplete step has no latency."""
        from metrics import StepMetric

        step = StepMetric(name="test", started_at=time.time())
        self.assertIsNone(step.latency_ms)
        self.assertFalse(step.is_complete)

    def test_step_metric_to_dict(self):
        """Test step metric serialization."""
        from metrics import StepMetric

        step = StepMetric(name="test_step", started_at=100.0, ended_at=100.5, ok=True)
        step.extra["items"] = 42

        result = step.to_dict()
        self.assertEqual(result["name"], "test_step")
        self.assertEqual(result["latency_ms"], 500.0)
        self.assertTrue(result["ok"])
        self.assertEqual(result["extra"]["items"], 42)


class TestRunMetrics(unittest.TestCase):
    """Test RunMetrics functionality."""

    def test_track_step_context_manager(self):
        """Test step tracking with context manager."""
        from metrics import RunMetrics

        metrics = RunMetrics(run_id="test-run")

        with metrics.track_step("step1") as step:
            step.extra["count"] = 10
            time.sleep(0.01)

        self.assertEqual(len(metrics.steps), 1)
        self.assertTrue(metrics.steps[0].ok)
        self.assertEqual(metrics.steps[0].extra["count"], 10)
        self.assertIsNotNone(metrics.steps[0].latency_ms)

    def test_track_step_captures_exception(self):
        """Test that exceptions are captured in step metrics."""
        from metrics import RunMetrics

        metrics = RunMetrics(run_id="test-run")

        with self.assertRaises(ValueError):
            with metrics.track_step("failing_step"):
                raise ValueError("Test error")

        self.assertEqual(len(metrics.steps), 1)
        self.assertFalse(metrics.steps[0].ok)
        self.assertIn("Test error", metrics.steps[0].error)

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        from metrics import RunMetrics

        metrics = RunMetrics(run_id="test-run")

        # 3 successful, 1 failed
        for i in range(3):
            with metrics.track_step(f"step{i}"):
                pass

        try:
            with metrics.track_step("failing"):
                raise Exception("fail")
        except:
            pass

        self.assertAlmostEqual(metrics.success_rate, 0.75)

    def test_failed_steps_property(self):
        """Test failed_steps property returns only failures."""
        from metrics import RunMetrics

        metrics = RunMetrics(run_id="test-run")

        with metrics.track_step("success"):
            pass

        try:
            with metrics.track_step("failure"):
                raise Exception("fail")
        except:
            pass

        failed = metrics.failed_steps
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].name, "failure")

    def test_mark_degraded(self):
        """Test marking features as degraded."""
        from metrics import RunMetrics

        metrics = RunMetrics(run_id="test-run")
        metrics.mark_degraded("bm25", "Index not available")

        self.assertEqual(len(metrics.degraded_features), 1)
        self.assertIn("bm25", metrics.degraded_features[0])


class TestMetricsCollector(unittest.TestCase):
    """Test MetricsCollector functionality."""

    def test_record_and_retrieve_stats(self):
        """Test recording and retrieving run statistics."""
        from metrics import MetricsCollector, RunMetrics

        collector = MetricsCollector(max_history=10)

        for i in range(3):
            run = RunMetrics(run_id=f"run-{i}")
            run.finish()
            collector.record(run)

        self.assertEqual(collector.run_count, 3)

    def test_max_history_limit(self):
        """Test that max_history is respected."""
        from metrics import MetricsCollector, RunMetrics

        collector = MetricsCollector(max_history=5)

        for i in range(10):
            run = RunMetrics(run_id=f"run-{i}")
            run.finish()
            collector.record(run)

        self.assertEqual(collector.run_count, 5)

    def test_average_latency(self):
        """Test average latency calculation."""
        from metrics import MetricsCollector, RunMetrics

        collector = MetricsCollector()

        for i in range(3):
            run = RunMetrics(run_id=f"run-{i}")
            time.sleep(0.01)
            run.finish()
            collector.record(run)

        avg = collector.average_latency_ms
        self.assertIsNotNone(avg)
        self.assertGreater(avg, 5)


# =============================================================================
# BM25 Index Tests
# =============================================================================

class TestBM25Tokenization(unittest.TestCase):
    """Test BM25 tokenization."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        from bm25_index import _tokenize

        tokens = _tokenize("Hello, World! This is a test.")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("test", tokens)
        self.assertIn("this", tokens)
        # Single character tokens should be filtered
        self.assertNotIn("a", tokens)

    def test_tokenize_removes_punctuation(self):
        """Test that punctuation is removed."""
        from bm25_index import _tokenize

        tokens = _tokenize("test@email.com, user's data!")
        self.assertNotIn("@", tokens)
        self.assertNotIn(",", tokens)
        self.assertNotIn("!", tokens)
        self.assertNotIn("'", tokens)

    def test_tokenize_empty_string(self):
        """Test tokenizing empty string."""
        from bm25_index import _tokenize

        tokens = _tokenize("")
        self.assertEqual(tokens, [])

    def test_tokenize_preserves_numbers(self):
        """Test that numbers are preserved."""
        from bm25_index import _tokenize

        tokens = _tokenize("version 2.0 released in 2024")
        self.assertIn("version", tokens)
        self.assertIn("2024", tokens)


class TestBM25Index(unittest.TestCase):
    """Test BM25Index core functionality."""

    def test_add_document(self):
        """Test adding documents to index."""
        from bm25_index import BM25Index

        index = BM25Index()
        added = index.add_document("doc1", ["hello", "world"])

        self.assertTrue(added)
        self.assertEqual(len(index), 1)
        self.assertIn("doc1", index.doc_id_set)

    def test_add_duplicate_rejected(self):
        """Test that duplicate documents are rejected."""
        from bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", ["hello", "world"])
        added = index.add_document("doc1", ["different", "content"])

        self.assertFalse(added)
        self.assertEqual(len(index), 1)

    def test_search_returns_ranked_results(self):
        """Test that search returns properly ranked results."""
        from bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", ["python", "programming", "language"])
        index.add_document("doc2", ["java", "programming", "language"])
        index.add_document("doc3", ["python", "snake", "animal"])

        results = index.search(["python"], top_k=3)

        self.assertGreater(len(results), 0)
        doc_ids = [doc_id for doc_id, _ in results]
        # Both doc1 and doc3 contain "python"
        self.assertTrue("doc1" in doc_ids or "doc3" in doc_ids)

    def test_search_empty_query(self):
        """Test searching with empty query."""
        from bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", ["hello", "world"])

        results = index.search([], top_k=10)
        self.assertEqual(results, [])

    def test_search_no_matching_terms(self):
        """Test searching for non-existent terms."""
        from bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", ["hello", "world"])

        results = index.search(["nonexistent", "terms"], top_k=10)
        self.assertEqual(results, [])

    def test_remove_document(self):
        """Test removing a document from index."""
        from bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", ["hello", "world"])
        index.add_document("doc2", ["foo", "bar"])

        removed = index.remove_document("doc1")

        self.assertTrue(removed)
        self.assertEqual(len(index), 1)
        self.assertNotIn("doc1", index.doc_id_set)

    def test_remove_nonexistent_document(self):
        """Test removing non-existent document returns False."""
        from bm25_index import BM25Index

        index = BM25Index()
        removed = index.remove_document("nonexistent")
        self.assertFalse(removed)

    def test_idf_calculation(self):
        """Test IDF values are calculated correctly."""
        from bm25_index import BM25Index

        index = BM25Index()
        # "common" appears in 2 docs, "rare" in 1 doc
        index.add_document("doc1", ["common", "rare"])
        index.add_document("doc2", ["common", "other"])

        # IDF for rare term should be higher
        self.assertGreater(index.idf["rare"], index.idf["common"])


class TestPersistentBM25Index(unittest.TestCase):
    """Test PersistentBM25Index with file operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.temp_dir, "test_bm25")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_json(self):
        """Test saving and loading index with JSON format."""
        from bm25_index import PersistentBM25Index, BM25Config
        from unittest.mock import MagicMock

        config = BM25Config(index_path=self.index_path)
        store = MagicMock()

        # Create and populate index
        index = PersistentBM25Index(config, store)
        index.add_document("doc1", "hello world python programming")
        index.add_document("doc2", "foo bar baz test")
        index.save()

        # Verify JSON file was created
        json_path = os.path.join(self.temp_dir, "test_bm25.json.gz")
        self.assertTrue(os.path.exists(json_path))

        # Create new index instance and verify it loads
        index2 = PersistentBM25Index(config, store)
        self.assertEqual(len(index2), 2)
        self.assertIn("doc1", index2.index.doc_id_set)
        self.assertIn("doc2", index2.index.doc_id_set)

    def test_auto_save_threshold(self):
        """Test that auto-save triggers at threshold."""
        from bm25_index import PersistentBM25Index, BM25Config
        from unittest.mock import MagicMock

        config = BM25Config(index_path=self.index_path, auto_save_threshold=3)
        store = MagicMock()

        index = PersistentBM25Index(config, store)

        # Add documents to trigger auto-save
        for i in range(5):
            index.add_document(f"doc{i}", f"content number {i}")

        # Verify JSON file was created
        json_path = os.path.join(self.temp_dir, "test_bm25.json.gz")
        self.assertTrue(os.path.exists(json_path))

    def test_get_stats_shows_format(self):
        """Test that get_stats shows storage format."""
        from bm25_index import PersistentBM25Index, BM25Config
        from unittest.mock import MagicMock

        config = BM25Config(index_path=self.index_path)
        store = MagicMock()

        index = PersistentBM25Index(config, store)
        index.add_document("doc1", "test content")
        index.save()

        stats = index.get_stats()
        self.assertEqual(stats["storage_format"], "json.gz")
        self.assertIn(".json.gz", stats["index_path"])

    def test_clear_removes_json_file(self):
        """Test that clear removes JSON index file."""
        from bm25_index import PersistentBM25Index, BM25Config
        from unittest.mock import MagicMock

        config = BM25Config(index_path=self.index_path)
        store = MagicMock()

        index = PersistentBM25Index(config, store)
        index.add_document("doc1", "test content")
        index.save()

        json_path = os.path.join(self.temp_dir, "test_bm25.json.gz")
        self.assertTrue(os.path.exists(json_path))

        index.clear()
        self.assertFalse(os.path.exists(json_path))
        self.assertEqual(len(index), 0)


class TestBM25IndexSerialization(unittest.TestCase):
    """Test BM25Index to_dict and from_dict methods."""

    def test_to_dict_basic(self):
        """Test basic serialization to dictionary."""
        from bm25_index import BM25Index

        index = BM25Index(k1=1.5, b=0.75)
        index.add_document("doc1", ["hello", "world"])
        index.add_document("doc2", ["foo", "bar"])

        data = index.to_dict()

        self.assertEqual(data["version"], 2)
        self.assertEqual(data["doc_ids"], ["doc1", "doc2"])
        self.assertEqual(data["doc_tokens"], [["hello", "world"], ["foo", "bar"]])
        self.assertEqual(data["k1"], 1.5)
        self.assertEqual(data["b"], 0.75)

    def test_from_dict_basic(self):
        """Test basic deserialization from dictionary."""
        from bm25_index import BM25Index

        data = {
            "version": 2,
            "doc_ids": ["doc1", "doc2"],
            "doc_tokens": [["hello", "world"], ["foo", "bar"]],
            "k1": 1.2,
            "b": 0.8,
        }

        index = BM25Index.from_dict(data)

        self.assertEqual(len(index), 2)
        self.assertIn("doc1", index.doc_id_set)
        self.assertIn("doc2", index.doc_id_set)
        self.assertEqual(index.k1, 1.2)
        self.assertEqual(index.b, 0.8)
        # Verify computed values are rebuilt
        self.assertFalse(index.needs_rebuild)
        self.assertGreater(len(index.idf), 0)

    def test_roundtrip_serialization(self):
        """Test that serialization round-trip preserves data."""
        from bm25_index import BM25Index

        # Create original index
        original = BM25Index(k1=1.8, b=0.6)
        original.add_document("doc1", ["python", "programming", "language"])
        original.add_document("doc2", ["java", "programming", "enterprise"])
        original.add_document("doc3", ["python", "data", "science"])

        # Serialize and deserialize
        data = original.to_dict()
        restored = BM25Index.from_dict(data)

        # Verify structure
        self.assertEqual(len(restored), len(original))
        self.assertEqual(set(restored.doc_ids), set(original.doc_ids))
        self.assertEqual(restored.k1, original.k1)
        self.assertEqual(restored.b, original.b)

        # Verify search produces same results
        query = ["python"]
        original_results = original.search(query, top_k=3)
        restored_results = restored.search(query, top_k=3)

        self.assertEqual(len(original_results), len(restored_results))
        for (orig_id, orig_score), (rest_id, rest_score) in zip(original_results, restored_results):
            self.assertEqual(orig_id, rest_id)
            self.assertAlmostEqual(orig_score, rest_score, places=5)

    def test_from_dict_empty_index(self):
        """Test deserializing empty index."""
        from bm25_index import BM25Index

        data = {
            "version": 2,
            "doc_ids": [],
            "doc_tokens": [],
            "k1": 1.5,
            "b": 0.75,
        }

        index = BM25Index.from_dict(data)
        self.assertEqual(len(index), 0)
        self.assertEqual(index.avgdl, 0.0)

    def test_from_dict_legacy_format(self):
        """Test deserializing legacy format without version."""
        from bm25_index import BM25Index

        data = {
            "doc_ids": ["doc1"],
            "doc_tokens": [["test", "content"]],
            "k1": 1.5,
            "b": 0.75,
        }

        index = BM25Index.from_dict(data)
        self.assertEqual(len(index), 1)
        self.assertIn("doc1", index.doc_id_set)

    def test_json_file_format(self):
        """Test actual JSON file format is correct."""
        import gzip
        import json
        from bm25_index import BM25Index

        index = BM25Index()
        index.add_document("doc1", ["hello", "world"])

        data = index.to_dict()

        # Verify it's valid JSON
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        self.assertEqual(parsed["doc_ids"], ["doc1"])
        self.assertEqual(parsed["doc_tokens"], [["hello", "world"]])


# =============================================================================
# Conversation Tests
# =============================================================================

class TestConversationTurn(unittest.TestCase):
    """Test ConversationTurn functionality."""

    def test_turn_creation(self):
        """Test creating a conversation turn."""
        from conversation import ConversationTurn

        turn = ConversationTurn(
            turn_id="turn-1",
            role="user",
            content="Hello!",
            timestamp=time.time(),
        )

        self.assertEqual(turn.role, "user")
        self.assertEqual(turn.content, "Hello!")

    def test_turn_to_dict(self):
        """Test turn serialization."""
        from conversation import ConversationTurn

        turn = ConversationTurn(
            turn_id="turn-1",
            role="assistant",
            content="Hi there!",
            timestamp=1000.0,
            metadata={"key": "value"}
        )

        result = turn.to_dict()
        self.assertEqual(result["turn_id"], "turn-1")
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["metadata"]["key"], "value")

    def test_turn_from_dict(self):
        """Test turn deserialization."""
        from conversation import ConversationTurn

        data = {
            "turn_id": "turn-1",
            "role": "user",
            "content": "Test",
            "timestamp": 1000.0,
        }

        turn = ConversationTurn.from_dict(data)
        self.assertEqual(turn.turn_id, "turn-1")
        self.assertEqual(turn.content, "Test")


class TestConversation(unittest.TestCase):
    """Test Conversation functionality."""

    def test_add_turn(self):
        """Test adding turns to conversation."""
        from conversation import Conversation

        conv = Conversation(conversation_id="conv-1")
        turn = conv.add_turn("user", "Hello!")

        self.assertEqual(len(conv), 1)
        self.assertEqual(turn.role, "user")
        self.assertEqual(turn.content, "Hello!")

    def test_get_recent_turns(self):
        """Test getting recent turns."""
        from conversation import Conversation

        conv = Conversation(conversation_id="conv-1")
        for i in range(5):
            conv.add_turn("user" if i % 2 == 0 else "assistant", f"Message {i}")

        recent = conv.get_recent_turns(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0].content, "Message 2")
        self.assertEqual(recent[2].content, "Message 4")

    def test_get_history_text(self):
        """Test formatting conversation history."""
        from conversation import Conversation

        conv = Conversation(conversation_id="conv-1")
        conv.add_turn("user", "What is AI?")
        conv.add_turn("assistant", "AI stands for Artificial Intelligence.")

        history = conv.get_history_text(max_turns=2)

        self.assertIn("User:", history)
        self.assertIn("Assistant:", history)
        self.assertIn("AI", history)

    def test_conversation_serialization(self):
        """Test conversation to/from dict."""
        from conversation import Conversation

        conv = Conversation(conversation_id="conv-1")
        conv.add_turn("user", "Test message")

        data = conv.to_dict()
        restored = Conversation.from_dict(data)

        self.assertEqual(restored.conversation_id, "conv-1")
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.turns[0].content, "Test message")


# =============================================================================
# Document Ingestion Tests
# =============================================================================

class TestChunkSplitter(unittest.TestCase):
    """Test ChunkSplitter functionality."""

    def test_split_short_text(self):
        """Test that short text is not split."""
        from ingest import ChunkSplitter

        splitter = ChunkSplitter(chunk_size=100)
        chunks = splitter.split("Short text")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Short text")

    def test_split_long_text(self):
        """Test splitting long text."""
        from ingest import ChunkSplitter

        splitter = ChunkSplitter(chunk_size=50, chunk_overlap=10)
        text = "This is a longer text that should be split into multiple chunks for processing."

        chunks = splitter.split(text)

        self.assertGreater(len(chunks), 1)
        # All chunks should be non-empty
        for chunk in chunks:
            self.assertTrue(len(chunk) > 0)

    def test_split_respects_separator(self):
        """Test that split tries to break at separator."""
        from ingest import ChunkSplitter

        splitter = ChunkSplitter(chunk_size=30, chunk_overlap=5, separator=" ")
        text = "Word1 Word2 Word3 Word4 Word5 Word6"

        chunks = splitter.split(text)

        # Chunks should break at word boundaries
        for chunk in chunks:
            # Should not break mid-word
            self.assertFalse(chunk.startswith("ord"))


class TestIngestHelpers(unittest.TestCase):
    """Test ingestion helper functions."""

    def test_iter_input_files_with_file(self):
        """Test iterating input files with a single file."""
        from ingest import iter_input_files

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            files = iter_input_files([temp_path])
            self.assertEqual(len(files), 1)
            self.assertEqual(str(files[0]), temp_path)
        finally:
            os.unlink(temp_path)

    def test_iter_input_files_with_directory(self):
        """Test iterating input files with a directory."""
        from ingest import iter_input_files

        temp_dir = tempfile.mkdtemp()
        try:
            # Create test files
            for i in range(3):
                with open(os.path.join(temp_dir, f"file{i}.txt"), "w") as f:
                    f.write(f"content {i}")

            files = iter_input_files([temp_dir])
            self.assertEqual(len(files), 3)
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    def test_iter_input_files_nonexistent_warns(self):
        """Test that nonexistent path is skipped with warning."""
        from ingest import iter_input_files

        files = iter_input_files(["/nonexistent/path"])
        self.assertEqual(len(files), 0)


class TestCleaningOptions(unittest.TestCase):
    """Test CleaningOptions functionality."""

    def test_from_config(self):
        """Test creating CleaningOptions from config."""
        from ingest import CleaningOptions
        from config import UnstructuredCleaningConfig

        config = UnstructuredCleaningConfig(
            enabled=True,
            extra_whitespace=True,
            lowercase=False,
        )

        opts = CleaningOptions.from_config(config)
        self.assertTrue(opts.enabled)
        self.assertTrue(opts.extra_whitespace)
        self.assertFalse(opts.lowercase)


# =============================================================================
# Agent Context Tests
# =============================================================================

class TestAgentContext(unittest.TestCase):
    """Test AgentContext functionality."""

    @classmethod
    def setUpClass(cls):
        """Skip tests if dependencies are missing."""
        try:
            from rag_agents import new_agent_context, AgentContext
            cls.new_agent_context = new_agent_context
            cls.AgentContext = AgentContext
            cls.skip_tests = False
        except ImportError as e:
            cls.skip_tests = True
            cls.skip_reason = str(e)

    def test_new_agent_context_creates_run_id(self):
        """Test that new context gets a unique run_id."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        ctx1 = self.new_agent_context("query1")
        ctx2 = self.new_agent_context("query2")

        self.assertIsNotNone(ctx1.run_id)
        self.assertIsNotNone(ctx2.run_id)
        self.assertNotEqual(ctx1.run_id, ctx2.run_id)

    def test_agent_context_stores_query(self):
        """Test that context stores original query."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        ctx = self.new_agent_context("test query", "conv-123")

        self.assertEqual(ctx.original_query, "test query")
        self.assertEqual(ctx.conversation_id, "conv-123")

    def test_add_warning(self):
        """Test adding warnings to context."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        ctx = self.new_agent_context("test")
        ctx.add_warning("Warning 1")
        ctx.add_warning("Warning 2")

        self.assertEqual(len(ctx.warnings), 2)
        self.assertIn("Warning 1", ctx.warnings)
        self.assertIn("Warning 2", ctx.warnings)


# =============================================================================
# JSON Parser Tests
# =============================================================================

class TestJSONParser(unittest.TestCase):
    """Test JSON parsing utilities."""

    @classmethod
    def setUpClass(cls):
        """Skip tests if dependencies are missing."""
        try:
            from llm_clients import JSONParser
            cls.JSONParser = JSONParser
            cls.skip_tests = False
        except ImportError as e:
            cls.skip_tests = True
            cls.skip_reason = str(e)

    def test_extract_json_from_markdown(self):
        """Test extracting JSON from markdown code blocks."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        text = '''Here is the result:
```json
{"key": "value", "number": 42}
```
Done!'''

        result = self.JSONParser.parse(text, default={})
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)

    def test_extract_raw_json_object(self):
        """Test extracting raw JSON object."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        text = 'The answer is {"key": "value"} as shown.'
        result = self.JSONParser.parse(text, default={})
        self.assertEqual(result, {"key": "value"})

    def test_extract_json_array(self):
        """Test extracting JSON array."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        text = 'Results: ["item1", "item2", "item3"]'
        result = self.JSONParser.parse(text, default=[], expected_type=list)
        self.assertEqual(result, ["item1", "item2", "item3"])

    def test_handle_trailing_commas(self):
        """Test handling trailing commas in JSON."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        text = '{"items": ["a", "b", "c",]}'
        result = self.JSONParser.parse(text, default={})
        self.assertEqual(result, {"items": ["a", "b", "c"]})

    def test_invalid_json_returns_default(self):
        """Test that invalid JSON returns default value."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        text = 'This is not valid JSON at all'
        result = self.JSONParser.parse(text, default={"fallback": True})
        self.assertEqual(result, {"fallback": True})

    def test_type_validation_dict(self):
        """Test type validation for expected dict."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        text = '["this", "is", "an", "array"]'
        result = self.JSONParser.parse(text, default={"default": True}, expected_type=dict)
        self.assertEqual(result, {"default": True})

    def test_type_validation_list(self):
        """Test type validation for expected list."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        text = '{"this": "is", "a": "dict"}'
        result = self.JSONParser.parse(text, default=["default"], expected_type=list)
        self.assertEqual(result, ["default"])


# =============================================================================
# StoredDoc Tests
# =============================================================================

class TestStoredDoc(unittest.TestCase):
    """Test StoredDoc functionality."""

    def test_equality_based_on_doc_id(self):
        """Test that StoredDoc equality is based on doc_id."""
        from redis_store import StoredDoc

        doc1 = StoredDoc(doc_id="abc", content="content1", meta={})
        doc2 = StoredDoc(doc_id="abc", content="different content", meta={"key": "value"})
        doc3 = StoredDoc(doc_id="xyz", content="content1", meta={})

        self.assertEqual(doc1, doc2)  # Same doc_id
        self.assertNotEqual(doc1, doc3)  # Different doc_id

    def test_hash_for_set_membership(self):
        """Test that StoredDoc can be used in sets."""
        from redis_store import StoredDoc

        doc1 = StoredDoc(doc_id="abc", content="test1", meta={})
        doc2 = StoredDoc(doc_id="abc", content="test2", meta={})
        doc3 = StoredDoc(doc_id="xyz", content="test3", meta={})

        doc_set = {doc1, doc2, doc3}
        self.assertEqual(len(doc_set), 2)  # doc1 and doc2 have same id

    def test_not_equal_to_non_stored_doc(self):
        """Test that StoredDoc is not equal to non-StoredDoc objects."""
        from redis_store import StoredDoc

        doc = StoredDoc(doc_id="abc", content="test", meta={})

        self.assertNotEqual(doc, "abc")
        self.assertNotEqual(doc, {"doc_id": "abc"})
        self.assertNotEqual(doc, None)


# =============================================================================
# Agent Registry Tests
# =============================================================================

class TestAgentRegistry(unittest.TestCase):
    """Test AgentRegistry functionality."""

    def test_register_agent(self):
        """Test registering an agent."""
        from agents_registry import AgentRegistry

        registry = AgentRegistry()

        def my_agent(query: str) -> str:
            return f"Result for: {query}"

        agent = registry.register(
            name="MyAgent",
            description="A test agent",
            callable=my_agent,
        )

        self.assertEqual(agent.name, "MyAgent")
        self.assertTrue(registry.has("MyAgent"))

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate name raises error."""
        from agents_registry import AgentRegistry

        registry = AgentRegistry()

        def agent1(x): return x

        registry.register("Agent", "First agent", callable=agent1)

        with self.assertRaises(ValueError):
            registry.register("Agent", "Duplicate", callable=agent1)

    def test_invoke_agent(self):
        """Test invoking a registered agent."""
        from agents_registry import AgentRegistry

        registry = AgentRegistry()

        def my_agent(query: str) -> str:
            return f"Result for: {query}"

        registry.register("MyAgent", "Test", callable=my_agent)
        result = registry.invoke("MyAgent", "test query")

        self.assertEqual(result, "Result for: test query")

    def test_get_nonexistent_raises_error(self):
        """Test that getting nonexistent agent raises KeyError."""
        from agents_registry import AgentRegistry

        registry = AgentRegistry()

        with self.assertRaises(KeyError):
            registry.get("NonexistentAgent")

    def test_get_optional_returns_none(self):
        """Test that get_optional returns None for nonexistent."""
        from agents_registry import AgentRegistry

        registry = AgentRegistry()
        result = registry.get_optional("NonexistentAgent")
        self.assertIsNone(result)

    def test_list_agents_by_category(self):
        """Test listing agents by category."""
        from agents_registry import AgentRegistry

        registry = AgentRegistry()

        def agent(x): return x

        registry.register("Agent1", "First", callable=agent, category="retrieval")
        registry.register("Agent2", "Second", callable=agent, category="retrieval")
        registry.register("Agent3", "Third", callable=agent, category="generation")

        retrieval_agents = registry.list_agents(category="retrieval")
        self.assertEqual(len(retrieval_agents), 2)

        generation_agents = registry.list_agents(category="generation")
        self.assertEqual(len(generation_agents), 1)

    def test_unregister_agent(self):
        """Test unregistering an agent."""
        from agents_registry import AgentRegistry

        registry = AgentRegistry()

        def agent(x): return x

        registry.register("Agent", "Test", callable=agent)
        self.assertTrue(registry.has("Agent"))

        removed = registry.unregister("Agent")
        self.assertTrue(removed)
        self.assertFalse(registry.has("Agent"))

    def test_find_by_tag(self):
        """Test finding agents by tag."""
        from agents_registry import AgentRegistry

        registry = AgentRegistry()

        def agent(x): return x

        registry.register("Agent1", "First", callable=agent, tags=["fast", "local"])
        registry.register("Agent2", "Second", callable=agent, tags=["slow", "remote"])
        registry.register("Agent3", "Third", callable=agent, tags=["fast", "remote"])

        fast_agents = registry.find_by_tag("fast")
        self.assertEqual(len(fast_agents), 2)

        local_agents = registry.find_by_tag("local")
        self.assertEqual(len(local_agents), 1)

    def test_to_dict(self):
        """Test exporting registry as dictionary."""
        from agents_registry import AgentRegistry

        registry = AgentRegistry()

        def agent(x): return x

        registry.register("Agent1", "First agent", callable=agent, category="test")

        result = registry.to_dict()
        self.assertEqual(result["total_agents"], 1)
        self.assertIn("Agent1", result["agents"])
        self.assertEqual(result["agents"]["Agent1"]["description"], "First agent")


# =============================================================================
# Redis Store Mock Tests
# =============================================================================

class TestRedisStoreMocked(unittest.TestCase):
    """Test RedisVectorStore with mocked Redis."""

    def setUp(self):
        """Set up mocked Redis client."""
        from config import RedisConfig, VectorIndexConfig

        self.config = RedisConfig(
            url="redis://localhost:6379/0",
            key_prefix="test",
            vector_index=VectorIndexConfig(name="test_vectors"),
        )

    @patch('redis.Redis.from_url')
    def test_ping_success(self, mock_from_url):
        """Test successful ping."""
        from redis_store import RedisVectorStore

        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_from_url.return_value = mock_client

        store = RedisVectorStore(self.config)
        result = store.ping()

        self.assertTrue(result)
        mock_client.ping.assert_called_once()

    @patch('redis.Redis.from_url')
    def test_ping_connection_error(self, mock_from_url):
        """Test ping with connection error."""
        from redis_store import RedisVectorStore
        import redis

        mock_client = MagicMock()
        mock_client.ping.side_effect = redis.ConnectionError("Connection refused")
        mock_from_url.return_value = mock_client

        store = RedisVectorStore(self.config)
        result = store.ping()

        self.assertFalse(result)

    @patch('redis.Redis.from_url')
    def test_make_doc_id_deterministic(self, mock_from_url):
        """Test that doc_id generation is deterministic."""
        from redis_store import RedisVectorStore

        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        store = RedisVectorStore(self.config)

        id1 = store.make_doc_id("content", {"key": "value"})
        id2 = store.make_doc_id("content", {"key": "value"})
        id3 = store.make_doc_id("different", {"key": "value"})

        self.assertEqual(id1, id2)  # Same content and meta
        self.assertNotEqual(id1, id3)  # Different content

    @patch('redis.Redis.from_url')
    def test_doc_key_format(self, mock_from_url):
        """Test document key format."""
        from redis_store import RedisVectorStore

        mock_client = MagicMock()
        mock_from_url.return_value = mock_client

        store = RedisVectorStore(self.config)
        key = store._doc_key("abc123")

        self.assertEqual(key, "test:doc:abc123")


# =============================================================================
# RRF Agent Tests
# =============================================================================

class TestRRFAgent(unittest.TestCase):
    """Test RRF (Reciprocal Rank Fusion) agent."""

    @classmethod
    def setUpClass(cls):
        """Skip tests if dependencies are missing."""
        try:
            from rag_agents import RRFAgent
            from redis_store import StoredDoc
            from config import RetrievalConfig
            cls.RRFAgent = RRFAgent
            cls.StoredDoc = StoredDoc
            cls.RetrievalConfig = RetrievalConfig
            cls.skip_tests = False
        except ImportError as e:
            cls.skip_tests = True
            cls.skip_reason = str(e)

    def test_rrf_fusion_basic(self):
        """Test basic RRF fusion of two result sets."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        config = self.RetrievalConfig(fused_top_k=5, rrf_k=60)
        agent = self.RRFAgent(config)

        doc1 = self.StoredDoc(doc_id="doc1", content="content1", meta={})
        doc2 = self.StoredDoc(doc_id="doc2", content="content2", meta={})
        doc3 = self.StoredDoc(doc_id="doc3", content="content3", meta={})

        # Run 1: doc1 ranked 1st, doc2 ranked 2nd
        run1 = [(doc1, 0.9), (doc2, 0.8)]
        # Run 2: doc2 ranked 1st, doc3 ranked 2nd
        run2 = [(doc2, 0.95), (doc3, 0.85)]

        results = agent.run([run1, run2])

        # doc2 appears in both runs, should be ranked higher
        doc_ids = [doc.doc_id for doc, _ in results]
        self.assertIn("doc2", doc_ids)
        # doc2 should be first since it appears in both
        self.assertEqual(doc_ids[0], "doc2")

    def test_rrf_empty_runs(self):
        """Test RRF with empty result sets."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        config = self.RetrievalConfig(fused_top_k=5, rrf_k=60)
        agent = self.RRFAgent(config)

        results = agent.run([[], []])
        self.assertEqual(results, [])

    def test_rrf_respects_top_k(self):
        """Test that RRF respects top_k limit."""
        if self.skip_tests:
            self.skipTest(f"Dependency missing: {self.skip_reason}")

        config = self.RetrievalConfig(fused_top_k=2, rrf_k=60)
        agent = self.RRFAgent(config)

        docs = [
            self.StoredDoc(doc_id=f"doc{i}", content=f"content{i}", meta={})
            for i in range(5)
        ]

        run1 = [(doc, 1.0 - i * 0.1) for i, doc in enumerate(docs)]

        results = agent.run([run1])
        self.assertEqual(len(results), 2)


# =============================================================================
# Text Report Tests
# =============================================================================

class TestTextReportBuilder(unittest.TestCase):
    """Test TextReportBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        from text_report import TextReportBuilder, ReportConfig
        
        self.config = ReportConfig(
            width=80,
            environment="test",
            user_role="tester",
            workspace="test-workspace",
        )
        self.builder = TextReportBuilder(self.config)

    def test_add_line(self):
        """Test adding lines to report."""
        self.builder._add_line("test line")
        self.assertEqual(self.builder._lines, ["test line"])

    def test_add_separator(self):
        """Test adding separator lines."""
        self.builder._add_separator("=", 10)
        self.assertEqual(self.builder._lines, ["=========="])

    def test_add_key_value(self):
        """Test adding key-value pairs."""
        self.builder._add_key_value("key", "value")
        self.assertIn("key", self.builder._lines[0])
        self.assertIn("value", self.builder._lines[0])

    def test_wrap_text(self):
        """Test text wrapping."""
        long_text = "a" * 100
        wrapped = self.builder._wrap_text(long_text)
        self.assertTrue(all(len(line) <= 80 for line in wrapped))

    def test_infer_agent_type(self):
        """Test agent type inference."""
        self.assertEqual(self.builder._infer_agent_type("PlannerAgent"), "planner")
        self.assertEqual(self.builder._infer_agent_type("DenseRetrievalAgent"), "retriever")
        self.assertEqual(self.builder._infer_agent_type("RerankingAgent"), "reranker")
        self.assertEqual(self.builder._infer_agent_type("AnswerSynthesisAgent"), "generator")
        self.assertEqual(self.builder._infer_agent_type("CriticAgent"), "classifier")
        self.assertEqual(self.builder._infer_agent_type("UnknownAgent"), "system")


class TestTextReportGeneration(unittest.TestCase):
    """Test text report generation functions."""

    def _create_mock_result(self):
        """Create a mock PipelineResult for testing."""
        # Create mock StepMetric
        @dataclass
        class MockStep:
            name: str
            started_at: float
            ended_at: float
            ok: bool = True
            error: Optional[str] = None
            extra: Dict[str, Any] = None
            
            def __post_init__(self):
                if self.extra is None:
                    self.extra = {}
            
            @property
            def latency_ms(self):
                return (self.ended_at - self.started_at) * 1000

        # Create mock RunMetrics
        @dataclass
        class MockMetrics:
            run_id: str
            started_at: float
            ended_at: float
            steps: List[MockStep]
            warnings: List[str] = None
            degraded_features: List[str] = None
            
            def __post_init__(self):
                if self.warnings is None:
                    self.warnings = []
                if self.degraded_features is None:
                    self.degraded_features = []
            
            @property
            def total_latency_ms(self):
                return (self.ended_at - self.started_at) * 1000
            
            @property
            def success_rate(self):
                if not self.steps:
                    return 1.0
                return sum(1 for s in self.steps if s.ok) / len(self.steps)

        # Create mock StoredDoc
        @dataclass
        class MockDoc:
            doc_id: str
            content: str
            meta: Dict[str, Any]

        # Create mock AgentContext
        @dataclass
        class MockContext:
            run_id: str
            original_query: str
            plan: Dict[str, Any]
            decomposed_queries: List[str]
            expansions: List[str]
            rewrites: List[Tuple[str, str]]
            dense_retrieved: List[Tuple[MockDoc, float]]
            bm25_retrieved: List[Tuple[MockDoc, float]]
            fused: List[Tuple[MockDoc, float]]
            auto_merged: List[Tuple[MockDoc, float]]
            reranked: List[Tuple[MockDoc, float]]
            critic_notes: List[Dict[str, Any]]
            warnings: List[str]

        # Create mock PipelineResult
        @dataclass
        class MockResult:
            answer: str
            context: MockContext
            metrics: MockMetrics
            success: bool = True
            error: Optional[str] = None

        now = time.time()
        
        steps = [
            MockStep("PlanningAgent", now, now + 0.1),
            MockStep("DenseRetrievalAgent", now + 0.1, now + 0.3),
            MockStep("BM25RetrievalAgent", now + 0.3, now + 0.4),
            MockStep("RerankingAgent", now + 0.4, now + 0.5),
            MockStep("AnswerSynthesisAgent", now + 0.5, now + 0.8),
        ]

        doc1 = MockDoc("doc1", "Test content 1", {"source_path": "/test/doc1.pdf", "page_number": 1})
        doc2 = MockDoc("doc2", "Test content 2", {"source_path": "/test/doc2.pdf", "page_number": 2})

        context = MockContext(
            run_id="test-run-123",
            original_query="What is the meaning of life?",
            plan={"use_decomposition": True, "use_rerank": True},
            decomposed_queries=["What is life?", "What is meaning?"],
            expansions=["life meaning purpose"],
            rewrites=[("original", "rewritten")],
            dense_retrieved=[(doc1, 0.95), (doc2, 0.85)],
            bm25_retrieved=[(doc1, 0.8), (doc2, 0.7)],
            fused=[(doc1, 0.9), (doc2, 0.8)],
            auto_merged=[(doc1, 0.9), (doc2, 0.8)],
            reranked=[(doc1, 0.95), (doc2, 0.85)],
            critic_notes=[{"quality": 0.9, "factual": True}],
            warnings=["Test warning"],
        )

        metrics = MockMetrics(
            run_id="test-run-123",
            started_at=now,
            ended_at=now + 1.0,
            steps=steps,
            warnings=[],
            degraded_features=[],
        )

        return MockResult(
            answer="The meaning of life is 42.",
            context=context,
            metrics=metrics,
            success=True,
        )

    def test_generate_text_report(self):
        """Test generating a text report."""
        from text_report import generate_text_report
        
        result = self._create_mock_result()
        report = generate_text_report(result)
        
        # Check report contains expected sections
        self.assertIn("AGENTIC RAG RUN REPORT", report)
        self.assertIn("USER QUERY", report)
        self.assertIn("FINAL ANSWER", report)
        self.assertIn("HIGH-LEVEL METRICS", report)
        self.assertIn("AGENT PLAN", report)
        self.assertIn("RETRIEVAL SUMMARY", report)

    def test_save_text_report(self):
        """Test saving a text report to file."""
        from text_report import save_text_report
        
        result = self._create_mock_result()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_report.txt")
            saved_path = save_text_report(result, filepath)
            
            self.assertTrue(os.path.exists(saved_path))
            
            with open(saved_path, "r") as f:
                content = f.read()
            
            self.assertIn("AGENTIC RAG RUN REPORT", content)

    def test_report_config_defaults(self):
        """Test ReportConfig default values."""
        from text_report import ReportConfig
        
        config = ReportConfig()
        
        self.assertEqual(config.width, 80)
        self.assertEqual(config.max_answer_preview, 2000)
        self.assertTrue(config.show_query_section)
        self.assertTrue(config.show_answer_section)
        self.assertTrue(config.show_metrics_section)


# =============================================================================
# Web Crawler Tests
# =============================================================================

class TestURLNormalizer(unittest.TestCase):
    """Test URL normalization."""

    def test_normalize_removes_fragment(self):
        """Test that fragments are removed."""
        from web_crawler import URLNormalizer
        
        normalized = URLNormalizer.normalize("https://example.com/page#section")
        self.assertEqual(normalized, "https://example.com/page")

    def test_normalize_lowercases_host(self):
        """Test that host is lowercased."""
        from web_crawler import URLNormalizer
        
        normalized = URLNormalizer.normalize("https://EXAMPLE.COM/Page")
        self.assertIn("example.com", normalized)

    def test_normalize_removes_default_ports(self):
        """Test that default ports are removed."""
        from web_crawler import URLNormalizer
        
        normalized = URLNormalizer.normalize("http://example.com:80/page")
        self.assertNotIn(":80", normalized)
        
        normalized = URLNormalizer.normalize("https://example.com:443/page")
        self.assertNotIn(":443", normalized)

    def test_normalize_trailing_slash(self):
        """Test trailing slash normalization."""
        from web_crawler import URLNormalizer
        
        normalized = URLNormalizer.normalize("https://example.com/page/")
        self.assertEqual(normalized, "https://example.com/page")
        
        # Root path keeps trailing slash
        normalized = URLNormalizer.normalize("https://example.com/")
        self.assertEqual(normalized, "https://example.com/")

    def test_get_domain(self):
        """Test domain extraction."""
        from web_crawler import URLNormalizer
        
        domain = URLNormalizer.get_domain("https://www.example.com/page")
        self.assertEqual(domain, "www.example.com")

    def test_is_same_domain(self):
        """Test same domain check."""
        from web_crawler import URLNormalizer
        
        self.assertTrue(
            URLNormalizer.is_same_domain(
                "https://example.com/page1",
                "https://example.com/page2"
            )
        )
        
        self.assertFalse(
            URLNormalizer.is_same_domain(
                "https://example.com/page",
                "https://other.com/page"
            )
        )


class TestLinkExtractor(unittest.TestCase):
    """Test link extraction from HTML."""

    def test_extract_links_absolute(self):
        """Test extracting absolute links."""
        from web_crawler import LinkExtractor
        
        html = '<a href="https://example.com/page">Link</a>'
        links = LinkExtractor.extract_links(html, "https://base.com")
        
        self.assertIn("https://example.com/page", links)

    def test_extract_links_relative(self):
        """Test extracting and resolving relative links."""
        from web_crawler import LinkExtractor
        
        html = '<a href="/page">Link</a>'
        links = LinkExtractor.extract_links(html, "https://example.com/dir/")
        
        self.assertIn("https://example.com/page", links)

    def test_extract_links_skips_javascript(self):
        """Test that javascript: links are skipped."""
        from web_crawler import LinkExtractor
        
        html = '<a href="javascript:void(0)">Link</a>'
        links = LinkExtractor.extract_links(html, "https://example.com")
        
        self.assertEqual(len(links), 0)

    def test_extract_links_skips_mailto(self):
        """Test that mailto: links are skipped."""
        from web_crawler import LinkExtractor
        
        html = '<a href="mailto:test@example.com">Email</a>'
        links = LinkExtractor.extract_links(html, "https://example.com")
        
        self.assertEqual(len(links), 0)

    def test_extract_title(self):
        """Test title extraction."""
        from web_crawler import LinkExtractor
        
        html = '<html><head><title>Test Page</title></head></html>'
        title = LinkExtractor.extract_title(html)
        
        self.assertEqual(title, "Test Page")

    def test_extract_title_missing(self):
        """Test title extraction when missing."""
        from web_crawler import LinkExtractor
        
        html = '<html><head></head></html>'
        title = LinkExtractor.extract_title(html)
        
        self.assertIsNone(title)


class TestCrawlResult(unittest.TestCase):
    """Test CrawlResult dataclass."""

    def test_is_html(self):
        """Test HTML content type detection."""
        from web_crawler import CrawlResult
        
        result = CrawlResult(
            url="https://example.com",
            content_type="text/html",
            content=b"<html></html>",
        )
        
        self.assertTrue(result.is_html)

    def test_is_not_html(self):
        """Test non-HTML content type detection."""
        from web_crawler import CrawlResult
        
        result = CrawlResult(
            url="https://example.com/doc.pdf",
            content_type="application/pdf",
            content=b"%PDF-1.4",
        )
        
        self.assertFalse(result.is_html)


class TestWebCrawlerConfig(unittest.TestCase):
    """Test WebCrawlerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from config import WebCrawlerConfig
        
        config = WebCrawlerConfig()
        
        self.assertEqual(config.max_depth, 2)
        self.assertEqual(config.max_pages, 100)
        self.assertTrue(config.same_domain_only)
        self.assertEqual(config.timeout, 30)
        self.assertEqual(config.delay, 0.5)
        self.assertTrue(config.verify_ssl)


class TestWebSearchConfig(unittest.TestCase):
    """Test WebSearchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from config import WebSearchConfig
        
        config = WebSearchConfig()
        
        self.assertFalse(config.enabled)  # Disabled by default
        self.assertEqual(config.max_results, 5)
        self.assertEqual(config.max_pages, 3)
        self.assertEqual(config.timeout, 15)
        self.assertTrue(config.include_in_synthesis)
        self.assertEqual(config.min_relevance, 0.3)
        self.assertTrue(config.cache_enabled)

    def test_trigger_keywords(self):
        """Test trigger keywords default list."""
        from config import WebSearchConfig
        
        config = WebSearchConfig()
        
        self.assertIn("latest", config.trigger_keywords)
        self.assertIn("recent", config.trigger_keywords)
        self.assertIn("current", config.trigger_keywords)
        self.assertIn("news", config.trigger_keywords)

    def test_blocked_domains(self):
        """Test blocked domains default list."""
        from config import WebSearchConfig
        
        config = WebSearchConfig()
        
        self.assertIn("facebook.com", config.blocked_domains)
        self.assertIn("twitter.com", config.blocked_domains)


class TestWebSearchAgent(unittest.TestCase):
    """Test WebSearchAgent logic without requiring full dependencies."""

    def test_should_search_trigger_keywords(self):
        """Test that trigger keywords logic works."""
        # Test the keyword matching logic directly
        trigger_keywords = ["latest", "recent", "current", "news"]
        
        def should_search(query, keywords):
            query_lower = query.lower()
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    return True
            return False
        
        # Should trigger on "latest"
        self.assertTrue(should_search("What are the latest developments?", trigger_keywords))
        
        # Should trigger on "recent"
        self.assertTrue(should_search("Show me recent news", trigger_keywords))
        
        # Should not trigger on normal query
        self.assertFalse(should_search("What is machine learning?", trigger_keywords))

    def test_should_search_plan_override(self):
        """Test that plan override logic works."""
        def should_search_with_plan(query, plan, trigger_keywords):
            if plan.get("use_web_search", False):
                return True
            query_lower = query.lower()
            for keyword in trigger_keywords:
                if keyword.lower() in query_lower:
                    return True
            return False
        
        trigger_keywords = ["latest", "recent"]
        
        # Plan explicitly enables web search
        plan = {"use_web_search": True}
        self.assertTrue(should_search_with_plan("What is ML?", plan, trigger_keywords))
        
        # Plan doesn't enable, no keywords
        plan = {"use_web_search": False}
        self.assertFalse(should_search_with_plan("What is ML?", plan, trigger_keywords))

    def test_extract_text_from_html_logic(self):
        """Test HTML text extraction logic."""
        import re
        
        def extract_text_from_html(html):
            # Remove script and style elements
            html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML comments
            html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
            
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", html)
            
            # Decode HTML entities
            text = text.replace("&nbsp;", " ")
            text = text.replace("&amp;", "&")
            
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text)
            
            return text.strip()
        
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <script>alert('test');</script>
                <p>Hello World</p>
                <style>.test { color: red; }</style>
                <div>More content</div>
            </body>
        </html>
        """
        
        text = extract_text_from_html(html)
        
        self.assertIn("Hello World", text)
        self.assertIn("More content", text)
        self.assertNotIn("alert", text)
        self.assertNotIn("color: red", text)

    def test_blocked_domain_check(self):
        """Test domain blocking logic."""
        blocked_domains = ["facebook.com", "twitter.com"]
        
        def is_blocked(url, blocked):
            for domain in blocked:
                if domain.lower() in url.lower():
                    return True
            return False
        
        self.assertTrue(is_blocked("https://facebook.com/page", blocked_domains))
        self.assertTrue(is_blocked("https://www.twitter.com/user", blocked_domains))
        self.assertFalse(is_blocked("https://wikipedia.org/wiki/Test", blocked_domains))


# =============================================================================
# Test Runner
# =============================================================================

def run_tests():
    """Run all tests with verbosity."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
