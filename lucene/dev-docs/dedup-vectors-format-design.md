<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements.  See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to You under the Apache License, Version 2.0
  (the "License"); you may not use this file except in compliance with
  the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# DedupFlatVectorsFormat — Design Document

## Overview

`DedupFlatVectorsFormat` is a `FlatVectorsFormat` that de-duplicates stored vectors on disk.
When multiple documents — or multiple fields — share the same vector, it is stored once in a
shared dictionary. Each field maintains a compact ordinal mapping from document ordinals to
dictionary ordinals.

## Index File Formats

### `.dvd` (Dedup Vector Data)

```
[CodecHeader]
[Dictionary regions: unique vectors stored contiguously per dim/encoding group]
  Group 1: vec_0, vec_1, ..., vec_{dictSize-1}   (each: dim × encoding.byteSize bytes)
[Per-field ordToDict mappings: vint-encoded arrays]
  Field 1: ordToDict[0], ordToDict[1], ..., ordToDict[size-1]
  Field 2: ...
[Per-field OrdToDoc DISI data (for sparse fields)]
[CodecFooter]
```

Dictionary vectors are stored contiguously, enabling O(1) random access via
`dictDataOffset + dictOrd × vectorByteSize`. Fields with the same dimension and encoding
share a single dictionary region (cross-field dedup).

### `.dvm` (Dedup Vector Meta)

```
[CodecHeader]
For each field:
  [int32]  field number
  [vint]   size (number of documents with vectors)
  [int32]  vector similarity function ordinal
  [vint]   dictSize (number of unique vectors in this field's dictionary)
  [vint]   dictDimension
  [vint]   dictEncoding ordinal (-1 if empty)
  [vlong]  dictDataOffset (byte offset into .dvd)
  [vlong]  mapOffset (byte offset into .dvd for ordToDict mapping)
  [vlong]  mapLength (byte length of ordToDict mapping)
  [OrdToDocDISI meta (see OrdToDocDISIReaderConfiguration)]
[int32]  -1 (end-of-fields sentinel)
[CodecFooter]
```

## Algorithms

### Hashing

Vectors are serialized to raw bytes once, then hashed in a single pass via `hashBytes()`.
The function computes two independent 32-bit hashes simultaneously over the byte array:
- Upper 32 bits: polynomial hash (matching `Arrays.hashCode` semantics on 4-byte groups)
- Lower 32 bits: FNV-1a (byte-level hash with good avalanche properties)

The single-pass design avoids iterating the vector data twice. The 64-bit output reduces
collision frequency compared to a single 32-bit hash, minimizing expensive read-back
verifications.

### Hash Collision Handling

Collisions are handled correctly — never silently. On hash hit:

1. The candidate vector is read back and compared byte-for-byte using bulk
   `readBytes()` + `Arrays.equals()` (JVM-vectorized).
2. If equal → reuse the existing `dictOrd` (true duplicate).
3. If not equal → linear-probe the hash space (`hash+1`, `hash+2`, ...) until a miss or
   match is found.

### Indexing (Flush Path)

Vectors are buffered in memory during indexing (same as the base format). At flush time:

1. `initDict` checks if the current dictionary can be reused (same dimension and encoding).
   If so, the hash map and dictionary temp file are preserved for cross-field dedup.
2. Each vector is serialized to raw bytes via the reusable `vectorBytes` buffer, then hashed
   and looked up in a `LongIntHashMap`.
3. Collision verification reads from the dict temp file, which requires toggling between
   read and write modes. This is acceptable during flush since the number of vectors is
   bounded by the indexing buffer size.
4. The `ordToDict` mapping is accumulated in an in-memory `int[]`.
5. At `finish()` time, dictionary temp files are bulk-copied into `.dvd`, followed by the
   ordToDict mappings (written from the in-memory arrays), followed by OrdToDoc DISI data.

### Merge (Two-Pass)

During merge, each field is processed in two passes to avoid the expensive dict temp file
toggling that would occur in a single-pass design.

**Pass 1** (`writeMergedVectorsToTemp`): Iterates `MergedVectorValues` and writes all
vectors (including duplicates) to a temp file. This is a simple sequential write with no
dedup logic. For `mergeOneFieldToIndex`, this temp file also serves as the HNSW scorer
source.

**Pass 2** (`dedupFromTempFile`): Reads back from the closed temp file and deduplicates
against the dictionary. This pass has two phases:

- **Phase A** (dict in read mode): Scans all vectors sequentially from the temp file. Each
  vector is hashed and looked up in the hash map. On hash hit, the candidate is verified
  byte-for-byte:
  - If the candidate was added by a previous field, it is read from the dict temp file.
  - If the candidate was added by the current field (intra-field dedup), it is read from
    the per-field temp file using a tracked index.
  New unique vectors are assigned dictOrds and their temp-file indices are recorded.

- **Phase B** (dict in write mode): Seeks to each new unique vector in the temp file and
  appends it to the dictionary. Only new vectors are read — duplicates are skipped entirely.

The dict temp file is toggled once per field (read for phase A, write for phase B), not per
vector. This eliminates the O(n²) worst-case IO of per-vector toggling.

**IO hints**: The HNSW scorer temp file is opened with `IOContext.DEFAULT.withHints(
FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM)` to enable optimal
memory-mapping for the random-access pattern of HNSW graph construction.

**Memory usage during merge**: `O(uniqueVectors × 16 bytes)` for the hash map, plus
`O(docsWithField × 4 bytes)` for the ordToDict array, plus a `DocsWithFieldSet` bitset per
field.

### Cross-Field Dedup

Fields with the same dimension and vector encoding share a single dictionary region. When
`initDict` is called for a new field, it checks if the current dictionary matches. If so,
the hash map and dictionary temp file are preserved, and vectors from the new field are
deduped against the existing dictionary.

Fields with different dimensions or encodings get separate dictionary regions, since the
dictionary must be contiguous for O(1) random access.

### Search Path

**Reader construction**: all `ordToDict` mappings are eagerly loaded from `.dvd` into `int[]`
arrays stored in the `FieldEntry`. This is a one-time cost at segment open.

**Identity case (no duplicates, `ordToDict` is null)**: the reader returns
`OffHeapFloatVectorValues` / `OffHeapByteVectorValues` directly — the exact same classes used
by `Lucene99FlatVectorsFormat`. The search hot path is identical to the base format with zero
overhead.

**Dedup case**: the reader returns `DedupFloatVectorValues` which implements `HasIndexSlice`
and overrides `ordToOffset(int ord)`:

```java
public long ordToOffset(int ord) {
    return (long) ordToDict[ord] * vectorByteSize;
}
```

This enables the `Lucene99MemorySegmentFlatVectorsScorer` to operate directly on the
memory-mapped dictionary region using SIMD intrinsics, without copying vectors to the heap.

### Off-Heap Scoring Integration

The `ordToOffset(int ord)` method was added to `KnnVectorValues` as a new public API with a
default implementation of `(long) ord * getVectorByteLength()` (contiguous layout). All
MemorySegment scorers were updated to call `values.ordToOffset(ord)` instead of hardcoding
`(long) ord * vectorByteSize`. This is backward-compatible: the default preserves existing
behavior for all formats.

### Off-Heap Size Reporting

`getOffHeapByteSize` reports `size × vectorByteSize` (total documents, not unique vectors).
Although the dictionary is smaller, the HNSW graph has one node per document, so the access
pattern spans the full ordinal range. Reporting the full size ensures Lucene's memory
preloading infrastructure correctly advises the OS to keep the relevant pages resident.

## Tradeoffs

| Aspect | Without Dedup | With Dedup (no duplicates) | With Dedup (duplicates) |
|--------|--------------|---------------------------|------------------------|
| Disk usage | 1× | 1× (+ small mapping overhead) | Reduced proportional to dedup ratio |
| Search latency | Baseline | Identical (delegates to OffHeap*VectorValues) | +1 int[] lookup per vector (off-heap SIMD preserved) |
| Index-time memory | Baseline | +hash map (~16B/unique vec) + mapping (4B/doc) | Same |
| Merge-time memory | Baseline | +hash map (~16B/unique vec) + mapping (4B/doc) | Same |
| Merge I/O | 1 pass | 2 passes over temp file + 1 dict toggle per field | Same |
| HNSW merge I/O | 1 pass + 1 temp file | 2 passes over temp file (reused as scorer) + 1 dict toggle | Same |

## Classes

| Class | Role |
|-------|------|
| `DedupFlatVectorsFormat` | Format entry point, SPI-registered |
| `DedupFlatVectorsWriter` | Two-pass dedup writer with cross-field dictionary |
| `DedupFlatVectorsReader` | Reader with eager ordToDict loading, identity-case delegation, HasIndexSlice support |
| `KnnVectorValues.ordToOffset` | New API enabling off-heap scoring with non-contiguous vector layouts |
