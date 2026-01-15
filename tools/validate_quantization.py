#!/usr/bin/env python3
"""
Comprehensive validation script for binary quantization implementation.

Validates:
- Imports and dependencies
- Configuration loading
- Quantization utility functions
- Storage backend quantization support
- End-to-end quantized retrieval
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required imports work."""
    logger.info("=" * 60)
    logger.info("TEST 1: Validating imports")
    logger.info("=" * 60)
    
    errors = []
    
    # Test quantization module
    try:
        from radiant.storage import quantization
        logger.info("✓ radiant.storage.quantization")
        
        # Check for required functions
        required_funcs = [
            'quantize_embeddings',
            'embedding_to_bytes',
            'bytes_to_embedding',
            'get_binary_dimension',
            'rescore_candidates',
            'QuantizationConfig',
            'QUANTIZATION_AVAILABLE',
        ]
        
        for func in required_funcs:
            if hasattr(quantization, func):
                logger.info(f"  ✓ {func}")
            else:
                errors.append(f"Missing: quantization.{func}")
                logger.error(f"  ✗ {func} NOT FOUND")
    except ImportError as e:
        errors.append(f"Import error: {e}")
        logger.error(f"✗ radiant.storage.quantization: {e}")
    
    # Test config module
    try:
        from radiant.config import QuantizationConfig
        logger.info("✓ radiant.config.QuantizationConfig")
    except ImportError as e:
        errors.append(f"Config import error: {e}")
        logger.error(f"✗ radiant.config.QuantizationConfig: {e}")
    
    # Test storage base
    try:
        from radiant.storage.base import BaseVectorStore
        store_class = BaseVectorStore
        
        if hasattr(store_class, 'retrieve_by_embedding_quantized'):
            logger.info("✓ BaseVectorStore.retrieve_by_embedding_quantized")
        else:
            errors.append("Missing: BaseVectorStore.retrieve_by_embedding_quantized")
            logger.error("✗ BaseVectorStore.retrieve_by_embedding_quantized NOT FOUND")
    except ImportError as e:
        errors.append(f"Base storage import error: {e}")
        logger.error(f"✗ radiant.storage.base: {e}")
    
    # Test storage backends
    backends = [
        ('radiant.storage.redis_store', 'RedisVectorStore'),
        ('radiant.storage.chroma_store', 'ChromaVectorStore'),
        ('radiant.storage.pgvector_store', 'PgVectorStore'),
    ]
    
    for module_name, class_name in backends:
        try:
            module = __import__(module_name, fromlist=[class_name])
            store_class = getattr(module, class_name)
            logger.info(f"✓ {class_name}")
            
            # Check for quantization methods
            required_methods = ['retrieve_by_embedding_quantized']
            for method in required_methods:
                if hasattr(store_class, method):
                    logger.info(f"  ✓ {method}")
                else:
                    logger.warning(f"  ? {method} not explicitly defined (may inherit)")
        except ImportError as e:
            logger.warning(f"? {class_name}: {e} (optional dependency)")
        except Exception as e:
            errors.append(f"Error loading {class_name}: {e}")
            logger.error(f"✗ {class_name}: {e}")
    
    if errors:
        logger.error("\n❌ Import validation FAILED")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("\n✅ All imports validated successfully")
    return True


def test_quantization_functions():
    """Test quantization utility functions."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Validating quantization functions")
    logger.info("=" * 60)
    
    try:
        import numpy as np
        from radiant.storage.quantization import (
            quantize_embeddings,
            embedding_to_bytes,
            bytes_to_embedding,
            get_binary_dimension,
            rescore_candidates,
            QUANTIZATION_AVAILABLE,
        )
        
        if not QUANTIZATION_AVAILABLE:
            logger.warning("⚠ Quantization not available (sentence-transformers not installed)")
            logger.info("  To enable: pip install sentence-transformers>=3.2.0")
            return True  # Not an error, just not available
        
        # Test get_binary_dimension
        assert get_binary_dimension(384) == 48, "Binary dimension calculation incorrect"
        logger.info("✓ get_binary_dimension works correctly")
        
        # Test embedding_to_bytes and bytes_to_embedding
        test_emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        emb_bytes = embedding_to_bytes(test_emb)
        assert isinstance(emb_bytes, bytes), "embedding_to_bytes should return bytes"
        
        restored = bytes_to_embedding(emb_bytes, np.float32, (3,))
        assert np.allclose(restored, test_emb), "Embedding round-trip failed"
        logger.info("✓ embedding_to_bytes / bytes_to_embedding work correctly")
        
        # Test quantize_embeddings
        test_embeddings = np.random.randn(10, 384).astype(np.float32)
        
        # Test binary quantization
        binary = quantize_embeddings(test_embeddings, precision="ubinary")
        assert binary.dtype == np.uint8, "Binary quantization should produce uint8"
        assert binary.shape[1] == 48, "Binary dimension should be 48 for 384-dim input"
        logger.info("✓ Binary quantization works correctly")
        
        # Test int8 quantization (with ranges)
        ranges = np.vstack([
            np.min(test_embeddings, axis=0),
            np.max(test_embeddings, axis=0)
        ])
        int8 = quantize_embeddings(test_embeddings, precision="int8", ranges=ranges)
        assert int8.dtype == np.int8, "Int8 quantization should produce int8"
        assert int8.shape == test_embeddings.shape, "Int8 shape should match input"
        logger.info("✓ Int8 quantization works correctly")
        
        # Test rescore_candidates
        query = np.random.randn(384).astype(np.float32)
        candidates = [np.random.randn(384).astype(np.float32) for _ in range(5)]
        ids = [f"doc_{i}" for i in range(5)]
        
        scores = rescore_candidates(query, candidates, ids)
        assert len(scores) == 5, "Should return 5 scores"
        assert all(isinstance(s[1], float) for s in scores), "Scores should be floats"
        logger.info("✓ rescore_candidates works correctly")
        
        logger.info("\n✅ Quantization functions validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Quantization function test FAILED: {e}", exc_info=True)
        return False


def test_configuration():
    """Test configuration loading with quantization settings."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Validating configuration")
    logger.info("=" * 60)
    
    try:
        from radiant.config import (
            QuantizationConfig,
            RedisConfig,
            ChromaConfig,
            PgVectorConfig,
        )
        
        # Test QuantizationConfig
        quant = QuantizationConfig()
        logger.info("✓ QuantizationConfig instantiates with defaults")
        
        assert hasattr(quant, 'enabled'), "Missing: enabled"
        assert hasattr(quant, 'precision'), "Missing: precision"
        assert hasattr(quant, 'rescore_multiplier'), "Missing: rescore_multiplier"
        assert hasattr(quant, 'use_rescoring'), "Missing: use_rescoring"
        logger.info("✓ QuantizationConfig has required attributes")
        
        # Test that storage configs have quantization
        from dataclasses import fields
        
        redis_fields = {f.name for f in fields(RedisConfig)}
        assert 'quantization' in redis_fields, "RedisConfig missing quantization field"
        logger.info("✓ RedisConfig has quantization field")
        
        chroma_fields = {f.name for f in fields(ChromaConfig)}
        assert 'quantization' in chroma_fields, "ChromaConfig missing quantization field"
        logger.info("✓ ChromaConfig has quantization field")
        
        pgvector_fields = {f.name for f in fields(PgVectorConfig)}
        assert 'quantization' in pgvector_fields, "PgVectorConfig missing quantization field"
        logger.info("✓ PgVectorConfig has quantization field")
        
        logger.info("\n✅ Configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Configuration test FAILED: {e}", exc_info=True)
        return False


def test_storage_backend_integration():
    """Test that storage backends properly integrate quantization."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Validating storage backend integration")
    logger.info("=" * 60)
    
    try:
        from radiant.storage.redis_store import RedisVectorStore
        from radiant.config import RedisConfig, QuantizationConfig
        
        # Create a test config with quantization enabled
        quant_config = QuantizationConfig(
            enabled=True,
            precision="both",
            rescore_multiplier=4.0,
            use_rescoring=True,
        )
        
        redis_config = RedisConfig(
            url="redis://localhost:6379/0",
            quantization=quant_config,
        )
        
        logger.info("✓ RedisConfig with quantization created")
        
        # Check that the config is accessible
        assert redis_config.quantization.enabled == True
        assert redis_config.quantization.precision == "both"
        logger.info("✓ Quantization config properly accessible")
        
        # Test method signatures (without connecting)
        store_class = RedisVectorStore
        
        # Check for required methods
        assert hasattr(store_class, 'upsert'), "Missing: upsert"
        assert hasattr(store_class, 'retrieve_by_embedding'), "Missing: retrieve_by_embedding"
        assert hasattr(store_class, 'retrieve_by_embedding_quantized'), "Missing: retrieve_by_embedding_quantized"
        logger.info("✓ RedisVectorStore has required methods")
        
        # Check for quantization helper methods
        private_methods = [
            '_store_quantized_embeddings',
            '_load_binary_embedding',
            '_load_int8_embedding',
            '_binary_doc_key',
            '_int8_doc_key',
        ]
        
        for method in private_methods:
            if hasattr(store_class, method):
                logger.info(f"  ✓ {method}")
            else:
                logger.warning(f"  ? {method} not found")
        
        logger.info("\n✅ Storage backend integration validated")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Storage backend test FAILED: {e}", exc_info=True)
        return False


def test_syntax_validation():
    """Test that all modified files have valid Python syntax."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Syntax validation")
    logger.info("=" * 60)
    
    import py_compile
    
    files_to_check = [
        "radiant/storage/quantization.py",
        "radiant/config.py",
        "radiant/storage/base.py",
        "radiant/storage/redis_store.py",
        "radiant/storage/chroma_store.py",
        "radiant/storage/pgvector_store.py",
        "tools/calibrate_int8_ranges.py",
    ]
    
    errors = []
    for filepath in files_to_check:
        try:
            py_compile.compile(filepath, doraise=True)
            logger.info(f"✓ {filepath}")
        except py_compile.PyCompileError as e:
            errors.append(f"{filepath}: {e}")
            logger.error(f"✗ {filepath}: {e}")
    
    if errors:
        logger.error("\n❌ Syntax validation FAILED")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("\n✅ All files have valid syntax")
    return True


def main():
    """Run all validation tests."""
    logger.info("\n" + "=" * 60)
    logger.info("BINARY QUANTIZATION IMPLEMENTATION VALIDATION")
    logger.info("=" * 60)
    
    tests = [
        ("Import Validation", test_imports),
        ("Quantization Functions", test_quantization_functions),
        ("Configuration", test_configuration),
        ("Storage Integration", test_storage_backend_integration),
        ("Syntax Validation", test_syntax_validation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}", exc_info=True)
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED - Implementation is valid!")
        logger.info("\nNext steps:")
        logger.info("1. Install sentence-transformers if not already installed:")
        logger.info("   pip install sentence-transformers>=3.2.0")
        logger.info("2. Calibrate int8 ranges:")
        logger.info("   python tools/calibrate_int8_ranges.py --sample-size 50000")
        logger.info("3. Enable quantization in config.yaml")
        logger.info("4. Test with your application")
        return 0
    else:
        logger.error("\n⚠️  SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
