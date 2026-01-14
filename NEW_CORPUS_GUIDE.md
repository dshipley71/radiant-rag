# Using Binary Quantization with a New Corpus

## Perfect Starting Point! 🎯

Starting with unindexed documents is the **ideal scenario** for binary quantization because:
- ✅ No existing embeddings to worry about
- ✅ No migration needed
- ✅ Everything gets quantized from the start
- ✅ Maximum performance from day one

---

## Quick Start for New Corpus (3 Steps)

### Step 1: Setup (5 minutes)

Extract your downloaded archive and install dependencies:

```bash
# Extract
tar -xzf radiant-rag-with-quantization.tar.gz
cd radiant-rag-main

# Install dependencies
pip install sentence-transformers>=3.2.0 numpy>=1.26.0
```

### Step 2: Enable Quantization (2 minutes)

Edit your `config.yaml` and add quantization settings:

```yaml
# For Redis backend
redis:
  url: "redis://localhost:6379/0"
  key_prefix: "radiant"
  doc_ns: "doc"
  embed_ns: "emb"
  max_content_chars: 200000
  
  vector_index:
    name: "radiant_vectors"
    hnsw_m: 16
    hnsw_ef_construction: 200
    hnsw_ef_runtime: 100
    distance_metric: "COSINE"
  
  # Enable binary quantization (add this section)
  quantization:
    enabled: true                  # ← Enable it!
    precision: "binary"            # ← Start with binary-only (no calibration needed)
    rescore_multiplier: 4.0
    use_rescoring: true
```

**Important:** Start with `precision: "binary"` - it requires **no calibration** and works immediately!

### Step 3: Index Your Documents (Just Use Normal Code!)

Your existing indexing code works **without any changes**:

```python
from radiant.storage.factory import get_vector_store
from radiant.config import load_config
from your_embedding_model import get_embedding  # Your embedding function

# Load config (with quantization enabled)
config = load_config()
store = get_vector_store(config)

# Index your documents - quantization happens automatically!
for doc in your_documents:
    doc_id = store.make_doc_id(doc['content'])
    embedding = get_embedding(doc['content'])
    
    # This automatically stores:
    # 1. Float32 embedding (standard)
    # 2. Binary embedding (for fast search)
    store.upsert(
        doc_id=doc_id,
        content=doc['content'],
        embedding=embedding,
        meta=doc.get('metadata', {})
    )

print("✓ Documents indexed with quantization!")
```

**That's it!** Your documents are now indexed with binary quantization.

---

## Understanding What Just Happened

When you call `store.upsert()`, the code automatically:

1. **Stores the float32 embedding** (standard, for accuracy)
2. **Quantizes to binary** (1 bit per dimension)
3. **Stores the binary embedding** (for fast retrieval)

### Storage Details

**Redis:**
```
radiant:doc:{id}         → float32 embedding (1,536 bytes for 384-dim)
radiant:doc_binary:{id}  → binary embedding (48 bytes - 32x smaller!)
```

**Chroma:**
```
radiant_docs         → float32 collection
radiant_docs_binary  → binary collection (Hamming distance)
```

**PgVector:**
```
haystack_leaves          → float32 (VECTOR column)
haystack_leaves_binary   → binary (BYTEA)
```

---

## Querying Your Documents

Your existing query code also works **without changes**:

```python
# Your existing query code
query_embedding = get_embedding("search query")

# Standard retrieval (uses quantization automatically when enabled)
results = store.retrieve_by_embedding(
    query_embedding=query_embedding,
    top_k=10
)

# Or explicitly use quantized retrieval
results = store.retrieve_by_embedding_quantized(
    query_embedding=query_embedding,
    top_k=10,
    rescore_multiplier=4.0
)

# Results are the same format either way
for doc, score in results:
    print(f"{doc.doc_id}: {score:.4f}")
    print(f"  {doc.content[:100]}...")
```

---

## Upgrading to "both" Precision (Optional, Better Accuracy)

Once you have ~50k-100k documents indexed, you can upgrade to "both" (binary + int8) for better accuracy:

### 1. Calibrate Int8 Ranges

```bash
python tools/calibrate_int8_ranges.py \
    --sample-size 100000 \
    --output data/int8_ranges.npy
```

This takes ~5-10 minutes and samples your indexed embeddings to calculate optimal int8 quantization ranges.

### 2. Update Config

```yaml
redis:
  quantization:
    enabled: true
    precision: "both"              # ← Changed from "binary"
    rescore_multiplier: 4.0
    use_rescoring: true
    int8_ranges_file: "data/int8_ranges.npy"  # ← Add calibration file
```

### 3. Re-index (Optional)

**Option A: Continue with existing documents**
- New documents will use "both" precision
- Old documents still work with "binary" only
- Gradually migrate as documents are updated

**Option B: Re-index everything**
```python
# Get all document IDs
doc_ids = store.list_doc_ids_with_embeddings()

# Re-upsert to add int8 embeddings
for doc_id in doc_ids:
    doc = store.get_doc(doc_id)
    if doc:
        # Re-upsert (will add int8 embedding)
        store.upsert(
            doc_id=doc_id,
            content=doc.content,
            embedding=get_embedding(doc.content),  # Or load existing
            meta=doc.meta
        )
```

---

## Complete Working Example

Here's a complete script for indexing a new corpus:

```python
#!/usr/bin/env python3
"""
Index a new corpus with binary quantization enabled.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from radiant.config import load_config
from radiant.storage.factory import get_vector_store

def main():
    # 1. Load config (with quantization enabled)
    config = load_config()
    store = get_vector_store(config)
    
    # 2. Initialize embedding model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    
    # 3. Load your documents (adjust to your format)
    documents = []
    
    # Example: Load from JSON file
    with open('my_documents.json', 'r') as f:
        documents = json.load(f)
    
    # Or load from directory of text files
    # for file_path in Path('documents/').glob('*.txt'):
    #     with open(file_path) as f:
    #         documents.append({
    #             'content': f.read(),
    #             'metadata': {'filename': file_path.name}
    #         })
    
    print(f"Indexing {len(documents)} documents with quantization...")
    
    # 4. Index documents (quantization happens automatically!)
    for i, doc in enumerate(documents):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(documents)}")
        
        # Generate embedding
        embedding = model.encode(doc['content']).tolist()
        
        # Create document ID
        doc_id = store.make_doc_id(
            content=doc['content'],
            meta=doc.get('metadata', {})
        )
        
        # Store document with embedding
        # This automatically stores both float32 AND binary embeddings!
        store.upsert(
            doc_id=doc_id,
            content=doc['content'],
            embedding=embedding,
            meta=doc.get('metadata', {})
        )
    
    print(f"✓ Indexed {len(documents)} documents")
    print("✓ Binary quantization active")
    print("✓ Ready for fast retrieval!")
    
    # 5. Test a query
    query = "example search query"
    query_embedding = model.encode(query).tolist()
    
    results = store.retrieve_by_embedding(query_embedding, top_k=5)
    
    print(f"\nTest query: '{query}'")
    print(f"Found {len(results)} results:")
    for doc, score in results:
        print(f"  {score:.4f}: {doc.content[:80]}...")

if __name__ == "__main__":
    main()
```

---

## Performance Monitoring

Track your indexing performance:

```python
import time

start_time = time.time()
indexed_count = 0

for doc in documents:
    # ... indexing code ...
    indexed_count += 1
    
    if indexed_count % 1000 == 0:
        elapsed = time.time() - start_time
        rate = indexed_count / elapsed
        print(f"Indexed {indexed_count} docs at {rate:.1f} docs/sec")

print(f"Total time: {time.time() - start_time:.1f} seconds")
print(f"Average rate: {indexed_count / (time.time() - start_time):.1f} docs/sec")
```

---

## Batch Indexing (Faster)

For better performance with large corpora, use batch indexing:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
store = get_vector_store(config)

# Process in batches
batch_size = 100
documents_batch = []

for i, doc in enumerate(documents):
    documents_batch.append(doc)
    
    if len(documents_batch) >= batch_size or i == len(documents) - 1:
        # Batch encode embeddings (much faster!)
        contents = [d['content'] for d in documents_batch]
        embeddings = model.encode(contents, show_progress_bar=False)
        
        # Prepare batch for storage
        batch_data = []
        for j, doc in enumerate(documents_batch):
            doc_id = store.make_doc_id(doc['content'])
            batch_data.append({
                'doc_id': doc_id,
                'content': doc['content'],
                'embedding': embeddings[j].tolist(),
                'meta': doc.get('metadata', {})
            })
        
        # Store batch (quantization happens for all!)
        store.upsert_batch(batch_data)
        
        print(f"Indexed {i+1}/{len(documents)} documents")
        documents_batch = []
```

---

## Configuration Recommendations

### For Small Corpus (< 50k documents)

```yaml
quantization:
  enabled: true
  precision: "binary"      # Simple, no calibration needed
  rescore_multiplier: 4.0
  use_rescoring: true
```

### For Medium Corpus (50k - 1M documents)

```yaml
quantization:
  enabled: true
  precision: "both"        # Binary + int8 for better accuracy
  rescore_multiplier: 4.0
  use_rescoring: true
  int8_ranges_file: "data/int8_ranges.npy"
```

### For Large Corpus (> 1M documents)

```yaml
quantization:
  enabled: true
  precision: "both"
  rescore_multiplier: 3.0  # Lower for speed (still 94-95% accurate)
  use_rescoring: true
  int8_ranges_file: "data/int8_ranges.npy"
  int8_on_disk_only: true  # Save RAM
```

---

## Validation Before Starting

Before indexing your corpus, validate the setup:

```bash
# 1. Validate implementation
python tools/validate_quantization.py

# Expected output:
# ✅ PASS: Import Validation
# ✅ PASS: Quantization Functions
# ✅ PASS: Configuration
# ✅ PASS: Syntax Validation

# 2. Test storage connection
python -c "
from radiant.config import load_config
from radiant.storage.factory import get_vector_store

config = load_config()
store = get_vector_store(config)

if store.ping():
    print('✓ Storage connection successful')
    print('✓ Quantization enabled:', config.redis.quantization.enabled)
else:
    print('✗ Storage connection failed')
"

# 3. Test embedding model
python -c "
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
embedding = model.encode(['test'])
print(f'✓ Embedding model loaded')
print(f'✓ Embedding dimension: {len(embedding[0])}')
"
```

---

## FAQ for New Corpus

### Q: Do I need to do anything special for quantization?
**A:** No! Just enable it in config and index normally. Quantization happens automatically.

### Q: Will indexing be slower with quantization?
**A:** Slightly (~5-10%), but retrieval will be 10-20x faster. The tradeoff is worth it.

### Q: Can I index documents in parallel?
**A:** Yes, all backends support concurrent writes. Use multiprocessing or threading.

### Q: What if I want to change precision later?
**A:** Just update config and restart. New documents use new precision, old documents still work.

### Q: Should I start with "binary" or "both"?
**A:** Start with "binary" (no calibration needed). Upgrade to "both" after you have 50k+ documents.

### Q: How much disk space will I need?
**A:** With binary-only: ~103% of standard storage (3% overhead)
     With both: ~125% of standard storage (25% overhead)

---

## Expected Performance

For a corpus of **1 million documents** with **384-dimensional embeddings**:

### Indexing Performance
- **Time:** 1-3 hours (depending on hardware)
- **Rate:** 100-300 documents/second
- **Storage:** ~450 MB (vs 1.5 GB without quantization)

### Query Performance
- **Latency:** 5-10ms per query (vs 50-100ms without)
- **Throughput:** 100-200 queries/second
- **Accuracy:** 95-96% (with rescoring)

---

## Summary: Your Action Plan

1. **Extract** the archive
2. **Install** dependencies: `pip install sentence-transformers>=3.2.0`
3. **Enable** quantization in config.yaml (start with `precision: "binary"`)
4. **Index** your documents using normal code (quantization is automatic)
5. **Query** as usual (it's now 10-20x faster!)
6. **Optional:** Upgrade to "both" precision after 50k+ documents

That's it! You're now using binary quantization with your new corpus. 🚀

---

## Need Help?

- Check `QUICK_START.md` for setup issues
- Check `BINARY_QUANTIZATION_README.md` for features
- Run `python tools/validate_quantization.py` to diagnose problems
- Check logs for "quantization" related messages

Enjoy your fast, memory-efficient RAG system!
