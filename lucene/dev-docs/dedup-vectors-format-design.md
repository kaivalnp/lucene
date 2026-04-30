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

The writer performs **no temporary-file IO**. Dictionary bytes are streamed directly into the
main `.dvd` output, and hash-collision verification reads from sources that are already
resident in memory or already-open segment readers.

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

Float vectors are hashed directly on their `Float.floatToRawIntBits` representation with no
intermediate `byte[]` materialization. Byte vectors hash their raw bytes. Both produce a
64-bit digest by combining two independent 32-bit hashes computed in a single pass:

- Upper 32 bits: polynomial hash (matching `Arrays.hashCode` semantics).
- Lower 32 bits: FNV-1a, with good avalanche.

Flush and merge don't need to produce compatible hashes for the same vector across paths: a
single writer instance processes one encoding per shared-dict group, so each group's hash
space is self-contained. The 64-bit output keeps collision frequency low enough that
byte-level verification is rare.

### Hash Collision Handling

Collisions are never silently coalesced. On hash hit, the candidate vector is compared
using typed-array equality (`Arrays.equals(float[], float[])` or `Arrays.equals(byte[],
byte[])`). A mismatch triggers linear probing (`hash+1`, `hash+2`, ...) until a miss or
match is found.

Where the candidate comes from depends on the path — see Flush and Merge below.

### Writing to `.dvd`

Unique dictionary vectors are written directly to the main `.dvd` output as they are
discovered; there is no temp file. For float vectors, a single reusable little-endian
`ByteBuffer` sized to one vector is rewritten in place and flushed via `writeBytes` — the
same pattern used by `Lucene99FlatVectorsWriter`. For byte vectors, the caller's buffer
goes straight through `writeBytes`. No per-vector `byte[]` allocation.

### Flush Path

Vectors are buffered in memory during indexing (same as the base format), so each field's
`FieldWriter.vectors` holds a pre-copied typed array for every doc throughout the writer's
lifetime. At flush time:

1. `initDict` checks whether the current shared-dict group can be reused (same dim + same
   encoding). If so, the hash map and `dictSources` tracking array are preserved; otherwise
   a new dict region starts at `data.getFilePointer()`.
2. For each doc, hash the typed array directly and probe `hashToOrd`.
3. On miss: assign the new target dict ord, append bytes to `data`, and record the
   on-heap source location as `(flushFieldIdx, inFieldOrd)` in `dictSources`.
4. On hash hit: fetch the candidate's on-heap typed array via `fields.get(candidateFieldIdx)
   .vectors.get(candidateInFieldOrd)` and compare with `Arrays.equals`. No disk read.

Collision verification never touches `.dvd` or any temp file — everything is in RAM.

### Merge Path

Merges walk each source segment once via `DocIDMerger`. Per-doc writes go straight into
`.dvd`; no intermediate temp file. There are three collision-resolution paths, depending on
where the current doc's vector comes from and where the candidate came from.

#### Sub types

One `DedupSub` per source segment that has vectors for the current field. Each sub carries
its own vector-values handle for iteration and a separate `verifyValues = values.copy()` so
random-access verification reads within the same sub cannot corrupt the iteration buffer.

A specialized `DedupAwareFloatSub` / `DedupAwareByteSub` is used when the source reader is
itself a `DedupFlatVectorsReader` (and the field has duplicates, i.e., `ordToDict != null`
in the source). In addition to the iteration handle, this sub holds:

- Borrowed `sourceOrdToDict` (direct reference into the source reader's `FieldEntry`).
- `sourceDictOrdCache` (size = source's `dictSize`, initialized to `-1`), populated lazily.
- `dictVerifyValues` — a dict-level `VectorValues` handle that reads directly by source
  dict ord (indexed into the source's dictionary slice), used only for cross-sub collision
  verification against this sub's contributions.

#### Generic per-doc path (non-dedup sources)

1. `current = sub.current()` — iteration buffer for the current source ord.
2. Hash, probe.
3. Miss: write to `data`, record `(mergeFieldIdx, subIdx, perDocOrd)` in `dictSources`.
4. Hit: resolve candidate via `dictSourceVector(existing)` (flush → on-heap typed array;
   merge → `sub.valueAt(sourceOrd)`). Compare with `Arrays.equals`. Continue probing on
   mismatch.

#### Dedup-aware lazy cache (dedup sources with duplicates)

1. `srcDictOrd = sub.sourceOrdToDict[iter.index()]`.
2. If `sub.sourceDictOrdCache[srcDictOrd] >= 0` → use it. **Zero hashing, zero byte
   comparison, zero IO.** This is the hot path: repeated duplicates within a source are
   resolved to their target dict ord with two array lookups.
3. Otherwise, read the vector, hash, probe. On hash hit, use the **same-sub shortcut**:
   - Compare the candidate's packed high 32 bits (`dictSources[existing] >>> 32`) against
     the current sub's packed key `(1 << 31) | (mergeFieldIdx << 16) | subIdx`.
   - Same sub → equality is decided purely by `(int) dictSources[existing] == srcDictOrd`:
     the source format guarantees distinct source dict ords correspond to distinct vectors,
     so no byte comparison is needed. Equal → return candidate's target dict ord; different
     → continue probing.
   - Cross-sub (different sub, or flush source) → full byte equality via
     `dictSourceVector(existing)`.
4. Populate `sourceDictOrdCache[srcDictOrd]` with the resolved target dict ord.

The cache ensures each unique source dict ord is hashed **at most once** per merge,
regardless of how many live docs reference it.

#### Dead-vector drop

Because the cache is populated lazily (on first live-doc encounter) rather than by walking
the source's dictionary up-front, source dict entries whose referencing docs have all been
deleted (via `liveDocs` or `DocMap`) are **never hashed, never written**. This prevents
dead vectors from accumulating across merge cycles — a pathology that would otherwise make
the `.dvd` grow monotonically even when live-vector counts stay flat.

### Cross-Field Dedup

Fields with the same dimension and vector encoding share a single dictionary region within
the writer's `.dvd`. When `initDict` is called for a new field, it checks whether the
current shared-dict group matches; if so, the hash map, `dictSources` tracking, and any
cached merge subs continue in use. Otherwise a new dict region begins at the current
`data.getFilePointer()`.

### Source tracking in `dictSources`

Per dict ord, a single `long` encodes where the dictionary vector came from, so collision
verification can reach it without any external state:

- **Flush** (bit 63 = 0): `(flushFieldIdx << 32) | inFieldOrd` — on-heap
  `fields.get(flushFieldIdx).vectors.get(inFieldOrd)`.
- **Merge** (bit 63 = 1): `(mergeFieldIdx << 48) | (subIdx << 32) | sourceOrd` where
  `sourceOrd` is the source dict ord for dedup-aware subs or the per-doc ord for generic
  subs. Resolution goes through `mergeFieldSubs.get(mergeFieldIdx).get(subIdx).valueAt(
  sourceOrd)`, and each sub type interprets `sourceOrd` according to its own semantics.

The high 32 bits serve as a packed sub identifier — two merge entries share these bits iff
they came from the same sub; a flush entry's top bit is 0, so it never matches a merge
entry. This is what the dedup-aware same-sub shortcut compares against.

### Reader construction / search

**Reader construction**: all `ordToDict` mappings are eagerly loaded from `.dvd` into
`int[]` arrays stored in the `FieldEntry`. One-time cost at segment open.

**Identity case (no duplicates, `ordToDict == null` in the field)**: the reader returns
`OffHeapFloatVectorValues` / `OffHeapByteVectorValues` directly — the exact same classes
used by `Lucene99FlatVectorsFormat`. The search hot path is identical to the base format
with zero overhead.

**Dedup case**: the reader returns `DedupFloatVectorValues` / `DedupByteVectorValues`,
which implement `HasIndexSlice` and override `ordToOffset(int ord)`:

```java
public long ordToOffset(int ord, int vectorByteSize) {
  return (long) ordToDict[ord] * vectorByteSize;
}
```

This enables `Lucene99MemorySegmentFlatVectorsScorer` to operate directly on the
memory-mapped dictionary region using SIMD intrinsics, without copying vectors to the heap.

### `getMergeInstance()` / `finishMerge()`

`DedupFlatVectorsReader` captures the original `IOContext` at construction (with
`FileTypeHint.DATA` + `FileDataHint.KNN_VECTORS`). `getMergeInstance()` switches to
`DataAccessHint.SEQUENTIAL` — the dedup writer walks each source segment mostly in forward
order via `DocIDMerger`, so readahead is a net win. Collision verification's occasional
random-access reads target recently-read vectors that are typically hot in the page cache.
`finishMerge()` reverts the hint for search.

`MergeState.knnVectorsReaders` already contains merge instances (Lucene calls
`getMergeInstance()` on them before handing them to the writer), so the writer doesn't call
it again.

### Off-Heap Scoring Integration

The `ordToOffset(int ord, int vectorByteSize)` method on `KnnVectorValues` is a public API
with a default of `(long) ord * vectorByteSize` (contiguous layout). All MemorySegment
scorers call `values.ordToOffset(ord, vectorByteSize)` instead of hardcoding the
multiplication. Backward-compatible: the default preserves existing behavior for all
formats.

### Off-Heap Size Reporting

`getOffHeapByteSize` reports `size × vectorByteSize` (total documents, not unique
vectors). Although the dictionary is smaller, the HNSW graph has one node per document, so
the access pattern spans the full ordinal range. Reporting the full size ensures Lucene's
memory-preloading infrastructure correctly advises the OS to keep the relevant pages
resident.

## Complexity

For a merged field with `N` total live docs, `U` unique vectors across all source
segments, and all sources written with `DedupFlatVectorsFormat`:

| Step | Cost |
|------|------|
| Per-doc cache lookup | O(1) per doc |
| Vector hashing | O(dim) per **unique source dict ord**, once per (sourceSub × srcDictOrd) |
| Byte-equality comparisons | Only on cross-sub hash collisions (rare with 64-bit hash) |
| `.dvd` writes | `U × dim × encoding.byteSize` bytes (dictionary) + `N × vint(dictOrd)` bytes (mapping) |
| Temp-file IO | None |

For non-dedup sources the per-doc path degrades gracefully to one hash + probe + possible
byte comparison per doc.

## Memory

Index-time working set:

- `LongIntHashMap` (hash → dict ord): ~16 bytes per unique vector in the current
  shared-dict group.
- `long[] dictSources`: 8 bytes per dict ord.
- `int[] ordToDict` per field (in-memory until `finish()`): 4 bytes per doc.
- Per-merge dedup-aware sub: `int[dictSize]` cache (4 bytes per source unique vector) +
  two `VectorValues` handles.
- Flush only: all `FieldWriter.vectors` stay on-heap through `finish()` (base format does
  the same).

## Tradeoffs

| Aspect | Without Dedup | With Dedup (no duplicates) | With Dedup (duplicates) |
|--------|--------------|---------------------------|------------------------|
| Disk usage | 1× | 1× (+ small mapping overhead) | Reduced proportional to dedup ratio |
| Search latency | Baseline | Identical (delegates to `OffHeap*VectorValues`) | +1 `int[]` lookup per vector (off-heap SIMD preserved) |
| Index-time memory | Baseline | +hash map (~16B/unique) + mapping (4B/doc) + `dictSources` (8B/unique) | Same |
| Merge-time memory | Baseline | As above + `int[dictSize]` cache per dedup-aware sub | Same |
| Merge IO | 1 pass + HNSW-temp | 1 pass, no temp file; dictionary appended in-line to `.dvd` | Same |
| Dead vectors across merges | N/A | Dropped (lazy cache only touches referenced source dict ords) | Same |

## Classes

| Class | Role |
|-------|------|
| `DedupFlatVectorsFormat` | Format entry point, SPI-registered |
| `DedupFlatVectorsWriter` | Single-pass dedup writer with direct-to-`data` dict, lazy dedup-aware merge cache |
| `DedupFlatVectorsReader` | Reader with eager ordToDict loading, identity-case delegation, HasIndexSlice support, SEQUENTIAL-hint merge instance |
| `DedupFlatVectorsWriter.DedupSub` / `FloatDedupSub` / `ByteDedupSub` | Generic per-segment merge sub, one iteration handle + one `copy()` verify handle |
| `DedupFlatVectorsWriter.DedupAwareFloatSub` / `DedupAwareByteSub` | Specialized merge sub for `DedupFlatVectorsReader` sources: lazy `sourceDictOrdCache`, dict-level verify handle, same-sub int-ord collision shortcut |
| `KnnVectorValues.ordToOffset` | API enabling off-heap scoring with non-contiguous vector layouts |
