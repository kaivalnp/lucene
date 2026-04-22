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
shared dictionary. Each field maintains a compact ordinal mapping from its document ordinals
to dictionary ordinals.

## Index File Formats

### `.dvd` (Dedup Vector Data)

```
[CodecHeader]
[Per-field dictionary regions: unique vectors stored contiguously]
  Field 1 dict: vec_0, vec_1, ..., vec_{uniqueCount-1}   (each: dim × encoding.byteSize bytes)
  Field 2 dict: vec_0, vec_1, ...                         (may share region with field 1)
[Per-field ordToDict mappings: vint-encoded arrays]
  Field 1 map: ordToDict[0], ordToDict[1], ..., ordToDict[size-1]
  Field 2 map: ...
[Per-field OrdToDoc DISI data (for sparse fields)]
[CodecFooter]
```

Dictionary vectors are stored contiguously per dictionary region, enabling O(1) random access
via `dictDataOffset + dictOrd × vectorByteSize`. Fields with the same dimension and encoding
share a single dictionary region (cross-field dedup).

### `.dvm` (Dedup Vector Meta)

```
[CodecHeader]
For each field:
  [int32]  field number
  [vint]   size (number of documents with vectors in this field)
  [int32]  vector similarity function ordinal
  [vint]   dictSize (number of unique vectors in this field's dictionary)
  [vint]   dictDimension
  [vint]   dictEncoding ordinal (-1 if empty)
  [vlong]  dictDataOffset (byte offset into .dvd for this field's dictionary)
  [vlong]  mapOffset (byte offset into .dvd for this field's ordToDict mapping)
  [vlong]  mapLength (byte length of the ordToDict mapping)
  [OrdToDocDISI meta (see OrdToDocDISIReaderConfiguration)]
[int32]  -1 (end-of-fields sentinel)
[CodecFooter]
```

## Algorithms

### Hashing

Vectors are hashed to a 64-bit value combining two independent 32-bit hashes:
- Upper 32 bits: `Arrays.hashCode` (Java's built-in polynomial hash)
- Lower 32 bits: FNV-1a (byte-level hash with good avalanche properties)

This reduces collision frequency compared to a single 32-bit hash. `Arrays.hashCode` alone
has known collisions for common vector patterns (e.g., `[0,0,0,0]` and `[2,2,2,2]` collide).
The FNV-1a component breaks these collisions. Fewer collisions means fewer expensive
read-back verifications during merge.

### Hash Collision Handling

Collisions are handled correctly — never silently. On hash hit:

1. The candidate vector is read back from the dictionary temp file and compared byte-for-byte.
2. If equal → reuse the existing `dictOrd` (true duplicate).
3. If not equal → linear-probe the hash space (`hash+1`, `hash+2`, ...) until a miss or
   match is found.

Read-back requires closing the dictionary temp file for writing, opening it for reading,
comparing, then reopening for writing (copying content to a new temp file). This is expensive
but happens only on hash collision, which is rare with 64-bit hashes.

### Indexing (Flush Path)

Vectors are buffered in memory during indexing (same as the base format). At flush time:

1. For each field, `initDict` checks if the current dictionary can be reused (same dimension
   and encoding). If so, the existing dictionary and hash map are preserved for cross-field
   dedup. Otherwise, a new dictionary is started.
2. Each vector is hashed and looked up in a `LongIntHashMap`.
   - **Hash miss**: the vector is new — written to the dictionary temp file, assigned a new
     `dictOrd`.
   - **Hash hit**: read-back verification as described above.
3. The `dictOrd` for each document is written to a mappings temp file.
4. At `finish()` time, dictionary temp files are copied into `.dvd`, followed by mappings,
   followed by OrdToDoc DISI data. Metadata is written to `.dvm`.

### Merge (Streaming Path)

During merge, vectors are streamed one at a time from `MergedVectorValues` — vector data is
never buffered in memory. The same hash-based dedup algorithm is used:

1. For each vector: compute hash → look up in hash map → verify on hit → assign dictOrd.
2. Dictionary vectors are written to a temp file as they are encountered.
3. The `ordToDict` mapping is written to a separate temp file.
4. Memory usage: `O(uniqueVectors × 12 bytes)` for the hash map, plus a `DocsWithFieldSet`
   bitset per field. For 1M unique vectors, this is ~12MB.

For `mergeOneFieldToIndex` (HNSW graph building), vectors are re-streamed to a second temp
file which is opened for random-access scoring via `OffHeapFloatVectorValues`.

### Cross-Field Dedup

Fields with the same dimension and vector encoding share a single dictionary region. When
`initDict` is called for a new field, it checks if the current dictionary matches. If so,
the hash map and dictionary temp file are preserved, and vectors from the new field are
deduped against the existing dictionary.

Fields with different dimensions or encodings get separate dictionary regions. This is
necessary because the dictionary must be contiguous for O(1) random access.

### Search Path

At reader construction time, all `ordToDict` mappings are eagerly loaded into `int[]` arrays
(one per field). This is a one-time cost at segment open.

**Identity case (no duplicates, `ordToDict` is null):** The reader returns
`OffHeapFloatVectorValues` / `OffHeapByteVectorValues` directly — the exact same classes used
by `Lucene99FlatVectorsFormat`. The search hot path is identical to the base format with zero
overhead.

**Dedup case:** The reader returns `DedupFloatVectorValues` which implements `HasIndexSlice`
and overrides `ordToOffset(int ord)`:

```java
public long ordToOffset(int ord) {
    return (long) ordToDict[ord] * vectorByteSize;
}
```

This enables the `Lucene99MemorySegmentFlatVectorsScorer` to operate directly on the
memory-mapped dictionary region using SIMD intrinsics, without copying vectors to the heap.
The scoring hot path becomes:

```java
// MemorySegment scorer calls ordToOffset → one int[] lookup
long addr = values.ordToOffset(node);
// Then operates directly on the memory-mapped dictionary via SIMD
vectorOp(seg, queryAddr, addr, dims);
```

### Off-Heap Scoring Integration

The `ordToOffset` method was added to `KnnVectorValues` as a new public API:

```java
public long ordToOffset(int ord) {
    return (long) ord * getVectorByteLength();  // default: contiguous layout
}
```

All MemorySegment scorers (`Lucene99MemorySegmentFloatVectorScorerSupplier`,
`Lucene99MemorySegmentFloatVectorScorer`, `Lucene99MemorySegmentByteVectorScorerSupplier`,
`Lucene99MemorySegmentByteVectorScorer`) were updated to call `values.ordToOffset(ord)`
instead of hardcoding `(long) ord * vectorByteSize`. This is a backward-compatible change:
the default implementation preserves the existing behavior for all existing formats.

## Tradeoffs

| Aspect | Without Dedup | With Dedup (no duplicates) | With Dedup (duplicates) |
|--------|--------------|---------------------------|------------------------|
| Disk usage | 1× | 1× (+ small mapping overhead) | Reduced proportional to dedup ratio |
| Search latency | Baseline | Identical (delegates to OffHeap*VectorValues) | +1 int[] lookup per vector access (off-heap SIMD scoring preserved) |
| Index-time memory | Baseline | +hash map (~12B/unique vec) | Same |
| Merge-time memory | Baseline | +hash map (~12B/unique vec) | Same |
| I/O pattern | Sequential | Sequential | Random (dictOrds not sequential) |

## Classes

| Class | Role |
|-------|------|
| `DedupFlatVectorsFormat` | Format entry point, SPI-registered |
| `DedupFlatVectorsWriter` | Streaming dedup writer with cross-field dictionary and collision handling |
| `DedupFlatVectorsReader` | Reader with eager ordToDict loading, identity-case delegation, HasIndexSlice support |
| `KnnVectorValues.ordToOffset` | New API enabling off-heap scoring with non-contiguous vector layouts |
