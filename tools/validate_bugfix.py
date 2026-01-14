#!/usr/bin/env python3
"""
Comprehensive verification script for bug fixes.

Validates:
1. Syntax validation of all modified files
2. Import validation
3. Agent calling convention fixes
4. AgentResult extraction fixes
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


def test_syntax_validation():
    """Test that modified files have valid syntax."""
    logger.info("="  * 60)
    logger.info("TEST 1: Syntax Validation")
    logger.info("=" * 60)
    
    import py_compile
    
    files_to_check = [
        "radiant/app.py",
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


def test_imports():
    """Test that all required imports work."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Import Validation")
    logger.info("=" * 60)
    
    errors = []
    
    # Test agent imports
    try:
        from radiant.agents import DenseRetrievalAgent, BM25RetrievalAgent, RRFAgent
        logger.info("✓ Agent imports successful")
    except ImportError as e:
        errors.append(f"Agent import error: {e}")
        logger.error(f"✗ Agent imports failed: {e}")
    
    # Test base agent
    try:
        from radiant.agents.base_agent import BaseAgent, AgentResult, AgentStatus
        logger.info("✓ BaseAgent imports successful")
    except ImportError as e:
        errors.append(f"BaseAgent import error: {e}")
        logger.error(f"✗ BaseAgent imports failed: {e}")
    
    # Test app module
    try:
        from radiant.app import RadiantRAG
        logger.info("✓ RadiantRAG import successful")
    except ImportError as e:
        errors.append(f"App import error: {e}")
        logger.error(f"✗ App import failed: {e}")
    
    if errors:
        logger.error("\n❌ Import validation FAILED")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("\n✅ All imports successful")
    return True


def test_agent_api():
    """Test that agent API is correctly understood."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Agent API Validation")
    logger.info("=" * 60)
    
    try:
        from radiant.agents.base_agent import BaseAgent
        import inspect
        
        # Check BaseAgent.run() signature
        run_sig = inspect.signature(BaseAgent.run)
        params = list(run_sig.parameters.keys())
        
        logger.info(f"BaseAgent.run() parameters: {params}")
        
        # Verify signature
        if params[0] != 'self':
            logger.error("✗ First parameter should be 'self'")
            return False
        
        if params[1] != 'correlation_id':
            logger.error("✗ Second parameter should be 'correlation_id'")
            return False
        
        if params[2] != 'kwargs':
            logger.error("✗ Third parameter should be 'kwargs' (**kwargs)")
            return False
        
        logger.info("✓ BaseAgent.run() signature correct")
        logger.info("  Expected: run(self, correlation_id=None, **kwargs)")
        logger.info("  This means all arguments except correlation_id must be keyword arguments")
        
        logger.info("\n✅ Agent API validation successful")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Agent API validation FAILED: {e}")
        return False


def test_agent_result_structure():
    """Test AgentResult structure."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: AgentResult Structure Validation")
    logger.info("=" * 60)
    
    try:
        from radiant.agents.base_agent import AgentResult, AgentStatus
        
        # Check required attributes
        required_attrs = ['data', 'success', 'status', 'error', 'warnings', 'metrics']
        
        for attr in required_attrs:
            if not hasattr(AgentResult, '__annotations__') or attr not in AgentResult.__annotations__:
                logger.error(f"✗ AgentResult missing attribute: {attr}")
                return False
            logger.info(f"  ✓ {attr}")
        
        logger.info("\n✅ AgentResult structure validated")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ AgentResult validation FAILED: {e}")
        return False


def test_code_patterns():
    """Test that code follows correct patterns."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Code Pattern Validation")
    logger.info("=" * 60)
    
    # Read app.py and check for correct patterns
    with open('radiant/app.py', 'r') as f:
        content = f.read()
    
    errors = []
    
    # Check that agent.run() calls use keyword arguments
    patterns_to_check = [
        ('dense_agent.run(query=query', 'DenseRetrievalAgent.run() uses keyword arguments'),
        ('bm25_agent.run(query=query', 'BM25RetrievalAgent.run() uses keyword arguments'),
        ('rrf_agent.run(runs=', 'RRFAgent.run() uses keyword arguments'),
        ('simple.run(query=query', 'SimplifiedOrchestrator.run() uses keyword arguments'),
    ]
    
    for pattern, description in patterns_to_check:
        if pattern in content:
            logger.info(f"✓ {description}")
        else:
            errors.append(f"Missing pattern: {pattern}")
            logger.error(f"✗ {description} - Pattern not found")
    
    # Check for AgentResult data extraction
    extraction_patterns = [
        ('dense_result.data if hasattr(dense_result', 'Dense result data extraction'),
        ('bm25_result.data if hasattr(bm25_result', 'BM25 result data extraction'),
        ('rrf_result.data if hasattr(rrf_result', 'RRF result data extraction'),
    ]
    
    for pattern, description in extraction_patterns:
        if pattern in content:
            logger.info(f"✓ {description}")
        else:
            errors.append(f"Missing pattern: {pattern}")
            logger.error(f"✗ {description} - Pattern not found")
    
    if errors:
        logger.error("\n❌ Code pattern validation FAILED")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("\n✅ All code patterns validated")
    return True


def main():
    """Run all validation tests."""
    logger.info("\n" + "=" * 60)
    logger.info("BUG FIX VALIDATION")
    logger.info("=" * 60)
    
    tests = [
        ("Syntax Validation", test_syntax_validation),
        ("Import Validation", test_imports),
        ("Agent API Validation", test_agent_api),
        ("AgentResult Structure", test_agent_result_structure),
        ("Code Pattern Validation", test_code_patterns),
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
        logger.info("\n🎉 ALL TESTS PASSED - Bug fixes are valid!")
        logger.info("\nNext steps:")
        logger.info("1. Test with actual search command:")
        logger.info("   python -m radiant search \"test query\"")
        logger.info("2. Verify results are returned (no TypeError)")
        logger.info("3. Test all search modes (dense, bm25, hybrid)")
        return 0
    else:
        logger.error("\n⚠️  SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
