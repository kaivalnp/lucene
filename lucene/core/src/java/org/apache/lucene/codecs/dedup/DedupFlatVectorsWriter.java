/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.codecs.dedup;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocIDMerger;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.internal.hppc.LongIntHashMap;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * Writer that de-duplicates vectors across all fields in a segment. Unique vectors are stored in a
 * contiguous dictionary region in the .dvd file. Each field stores a compact ordinal mapping from
 * document ords to dictionary ords.
 *
 * <p>This writer does no temporary-file IO and avoids materializing {@code byte[]} copies of float
 * vectors. Dictionary bytes are streamed directly into the main {@code .dvd} output; hash-
 * collision verification operates on typed arrays sourced from memory or already-open segment
 * readers:
 *
 * <ul>
 *   <li><b>Flush</b>: each field's vectors are already on-heap in {@link FieldWriter#vectors}. On a
 *       hash hit, the incoming typed vector is compared directly with the on-heap candidate via
 *       {@link Arrays#equals(float[], float[])} / {@link Arrays#equals(byte[], byte[])}.
 *   <li><b>Merge</b>: each field's per-segment readers (from {@link MergeState}) are kept cached in
 *       {@link #mergeFieldSubs} for the lifetime of the shared-dict group. On a hash hit the
 *       candidate is re-fetched from its source segment via a dedicated {@code verifyValues}
 *       handle, so random-access verification reads never alias the iteration buffer.
 *   <li><b>Merge, dedup-aware fast path</b>: when a source segment is itself written with {@link
 *       DedupFlatVectorsFormat} and the field has duplicates, per-doc iteration consults a lazy
 *       {@code srcDictOrd → targetDictOrd} cache on the sub. Each source dict ord is hashed at most
 *       once, regardless of how many live docs reference it — and dict entries whose docs have all
 *       been deleted are never touched, so dead vectors are naturally dropped at merge. Within the
 *       same sub, equality on hash collision is decided by comparing source dict ords (the source
 *       format guarantees distinct source dict ords correspond to distinct vectors); cross-sub
 *       collisions still fall back to full byte equality.
 * </ul>
 *
 * <p>Floats are hashed directly on their {@link Float#floatToRawIntBits(float)} representation with
 * no intermediate {@code byte[]} materialization for hashing or for the typed-array equality used
 * during collision verification. Writing floats to {@code data} uses a reusable little- endian
 * {@link java.nio.ByteBuffer} (same approach as {@code Lucene99FlatVectorsWriter}).
 *
 * @lucene.experimental
 */
final class DedupFlatVectorsWriter extends FlatVectorsWriter {

  private static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

  private final SegmentWriteState segmentWriteState;
  private final IndexOutput meta;
  private final IndexOutput data;

  private final List<FieldWriter<?>> fields = new ArrayList<>();
  private final List<PendingField> pendingFields = new ArrayList<>();

  // --- Dictionary state (shared across fields with matching dim/encoding) ---

  /** Hash map for dedup: 64-bit vector hash → dictOrd. */
  private final LongIntHashMap hashToOrd = new LongIntHashMap();

  /**
   * Merge-only: per-merge-field cached list of source-segment sub readers. Index is {@code
   * mergeFieldIdx}, as recorded in {@link #dictSources} for any dict ord contributed by the merge
   * path. Kept alive until the shared-dict group ends (different dim/encoding) or {@link #finish()}
   * / {@link #close()}.
   *
   * <p>The cached {@link DedupSub}s hold references to {@code MergeState}'s readers (they do not
   * own them); {@link MergeState} is responsible for closing its readers.
   */
  private final List<List<DedupSub<?>>> mergeFieldSubs = new ArrayList<>();

  /** Number of dict ords in the current shared-dict group. */
  private int dictSize;

  private long dictDataOffset;
  private int dictDimension;
  private VectorEncoding dictEncoding;
  private boolean dictInitialized;

  /**
   * Reusable little-endian {@link ByteBuffer} sized to one float vector (for the current shared-
   * dict group's dim). Rewritten in-place per write. Null for byte-encoded groups. Same approach as
   * {@code Lucene99FlatVectorsWriter.writeFloat32Vectors}.
   */
  private ByteBuffer floatWriteBuffer;

  /**
   * For each dictOrd in the current shared-dict group, the source-location descriptor needed to
   * re-fetch the vector during collision verification. Bit layout:
   *
   * <ul>
   *   <li>Flush (bit 63 = 0): {@code (flushFieldIdx << 32) | inFieldOrd}
   *   <li>Merge (bit 63 = 1): {@code (mergeFieldIdx << 48) | (subIdx << 32) | sourceOrd}
   * </ul>
   *
   * Sign bit distinguishes the two. {@code mergeFieldIdx} gets 15 bits and {@code subIdx} gets 16
   * bits — both are well under their limits in any realistic scenario.
   */
  private long[] dictSources;

  private boolean finished;

  DedupFlatVectorsWriter(SegmentWriteState state, FlatVectorsScorer scorer) throws IOException {
    super(scorer);
    this.segmentWriteState = state;

    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, DedupFlatVectorsFormat.META_EXTENSION);
    String dataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, DedupFlatVectorsFormat.DATA_EXTENSION);

    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      data = state.directory.createOutput(dataFileName, state.context);

      CodecUtil.writeIndexHeader(
          meta,
          DedupFlatVectorsFormat.META_CODEC_NAME,
          DedupFlatVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          data,
          DedupFlatVectorsFormat.DATA_CODEC_NAME,
          DedupFlatVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, this);
      throw t;
    }
  }

  @Override
  public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    FieldWriter<?> newField = FieldWriter.create(fieldInfo);
    fields.add(newField);
    return newField;
  }

  // ---- Flush path ----

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    for (int fieldIdx = 0; fieldIdx < fields.size(); fieldIdx++) {
      FieldWriter<?> field = fields.get(fieldIdx);
      flushField(field, fieldIdx, maxDoc, sortMap);
      field.finish();
    }
  }

  private <T> void flushField(FieldWriter<T> field, int fieldIdx, int maxDoc, Sorter.DocMap sortMap)
      throws IOException {
    List<T> vectors = field.vectors;
    DocsWithFieldSet docsWithField = field.docsWithField;
    int size = vectors.size();

    if (size == 0) {
      pendingFields.add(
          new PendingField(
              field.fieldInfo,
              0,
              maxDoc,
              docsWithField,
              null,
              0,
              0,
              field.fieldInfo.getVectorDimension(),
              field.fieldInfo.getVectorEncoding()));
      return;
    }

    initDict(field.fieldInfo);

    int[] ordMap = null;
    if (sortMap != null) {
      ordMap = new int[size];
      DocsWithFieldSet newDocsWithField = new DocsWithFieldSet();
      KnnVectorsWriter.mapOldOrdToNewOrd(docsWithField, sortMap, null, ordMap, newDocsWithField);
      docsWithField = newDocsWithField;
    }

    int[] ordToDict = new int[size];
    for (int newOrd = 0; newOrd < size; newOrd++) {
      int srcOrd = ordMap != null ? ordMap[newOrd] : newOrd;
      T vec = vectors.get(srcOrd);
      ordToDict[newOrd] = addFlushVector(fieldIdx, srcOrd, vec);
    }

    pendingFields.add(
        new PendingField(
            field.fieldInfo,
            size,
            maxDoc,
            docsWithField,
            ordToDict,
            dictSize,
            dictDataOffset,
            dictDimension,
            dictEncoding));
  }

  /**
   * Flush-path dedup. Incoming and candidate vectors are both typed arrays; comparison is a direct
   * {@link Arrays#equals} with no byte serialization on either side. On a miss, the vector is
   * written directly to {@code data} (without materializing an intermediate {@code byte[]}).
   */
  private <T> int addFlushVector(int fieldIdx, int inFieldOrd, T vec) throws IOException {
    long hash;
    if (vec instanceof float[] f) {
      hash = hashFloats(f);
    } else if (vec instanceof byte[] b) {
      hash = hashBytes(b, b.length);
    } else {
      throw new AssertionError("Unexpected vector type: " + vec.getClass());
    }

    for (int probe = 0; ; probe++) {
      long probeHash = hash + probe;
      int existing = hashToOrd.getOrDefault(probeHash, -1);
      if (existing < 0) {
        int dictOrd = dictSize++;
        hashToOrd.put(probeHash, dictOrd);
        writeVectorToData(vec);
        recordFlushSource(dictOrd, fieldIdx, inFieldOrd);
        return dictOrd;
      }
      if (typedVectorsEqual(vec, dictSourceVector(existing))) {
        return existing;
      }
      // Collision — continue probing.
    }
  }

  // ---- Merge path ----

  @Override
  public void mergeOneFlatVectorField(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException {
    initDict(fieldInfo);
    if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
      mergeFloatField(fieldInfo, mergeState);
    } else {
      mergeByteField(fieldInfo, mergeState);
    }
  }

  private void mergeFloatField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    int maxDoc = segmentWriteState.segmentInfo.maxDoc();
    int mergeFieldIdx = mergeFieldSubs.size();
    List<DedupSub<?>> subs = new ArrayList<>();
    mergeFieldSubs.add(subs);
    buildFloatSubs(subs, fieldInfo, mergeState);

    if (subs.isEmpty()) {
      pendingFields.add(
          new PendingField(
              fieldInfo,
              0,
              maxDoc,
              new DocsWithFieldSet(),
              null,
              dictSize,
              dictDataOffset,
              dictDimension,
              dictEncoding));
      return;
    }

    DocIDMerger<DedupSub<?>> merger = DocIDMerger.of(subs, mergeState.needsIndexSort);
    DocsWithFieldSet docsWithField = new DocsWithFieldSet();
    int[] ordToDictTmp = new int[16];
    int count = 0;

    for (DedupSub<?> sub = merger.next(); sub != null; sub = merger.next()) {
      int dictOrd;
      if (sub instanceof DedupAwareFloatSub das) {
        // Lazy cache keyed on source dict ord. Each source dict ord is hashed at most once per
        // merge. Dead dict entries (never referenced by any live doc) are never hashed — so the
        // target dict doesn't accumulate unreachable vectors across merge cycles.
        int srcDictOrd = das.sourceOrdToDict[sub.iteratorIndex()];
        int cached = das.sourceDictOrdCache[srcDictOrd];
        if (cached >= 0) {
          dictOrd = cached;
        } else {
          dictOrd = addDedupAwareFloatVector(das.current(), mergeFieldIdx, sub.subIdx, srcDictOrd);
          das.sourceDictOrdCache[srcDictOrd] = dictOrd;
        }
      } else {
        float[] current = ((FloatDedupSub) sub).current();
        dictOrd = addFloatMergeVector(current, mergeFieldIdx, sub.subIdx, sub.iteratorIndex());
      }
      if (count == ordToDictTmp.length) {
        ordToDictTmp = ArrayUtil.grow(ordToDictTmp);
      }
      ordToDictTmp[count++] = dictOrd;
      docsWithField.add(sub.mappedDocID);
    }

    int[] ordToDict = count == 0 ? null : ArrayUtil.copyOfSubArray(ordToDictTmp, 0, count);
    pendingFields.add(
        new PendingField(
            fieldInfo,
            count,
            maxDoc,
            docsWithField,
            ordToDict,
            dictSize,
            dictDataOffset,
            dictDimension,
            dictEncoding));
  }

  private void mergeByteField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    int maxDoc = segmentWriteState.segmentInfo.maxDoc();
    int mergeFieldIdx = mergeFieldSubs.size();
    List<DedupSub<?>> subs = new ArrayList<>();
    mergeFieldSubs.add(subs);
    buildByteSubs(subs, fieldInfo, mergeState);

    if (subs.isEmpty()) {
      pendingFields.add(
          new PendingField(
              fieldInfo,
              0,
              maxDoc,
              new DocsWithFieldSet(),
              null,
              dictSize,
              dictDataOffset,
              dictDimension,
              dictEncoding));
      return;
    }

    DocIDMerger<DedupSub<?>> merger = DocIDMerger.of(subs, mergeState.needsIndexSort);
    DocsWithFieldSet docsWithField = new DocsWithFieldSet();
    int[] ordToDictTmp = new int[16];
    int count = 0;

    for (DedupSub<?> sub = merger.next(); sub != null; sub = merger.next()) {
      int dictOrd;
      if (sub instanceof DedupAwareByteSub das) {
        int srcDictOrd = das.sourceOrdToDict[sub.iteratorIndex()];
        int cached = das.sourceDictOrdCache[srcDictOrd];
        if (cached >= 0) {
          dictOrd = cached;
        } else {
          dictOrd = addDedupAwareByteVector(das.current(), mergeFieldIdx, sub.subIdx, srcDictOrd);
          das.sourceDictOrdCache[srcDictOrd] = dictOrd;
        }
      } else {
        byte[] current = ((ByteDedupSub) sub).current();
        dictOrd = addByteMergeVector(current, mergeFieldIdx, sub.subIdx, sub.iteratorIndex());
      }
      if (count == ordToDictTmp.length) {
        ordToDictTmp = ArrayUtil.grow(ordToDictTmp);
      }
      ordToDictTmp[count++] = dictOrd;
      docsWithField.add(sub.mappedDocID);
    }

    int[] ordToDict = count == 0 ? null : ArrayUtil.copyOfSubArray(ordToDictTmp, 0, count);
    pendingFields.add(
        new PendingField(
            fieldInfo,
            count,
            maxDoc,
            docsWithField,
            ordToDict,
            dictSize,
            dictDataOffset,
            dictDimension,
            dictEncoding));
  }

  // ---- Per-vector merge helpers (shared between generic per-doc and dedup-aware lazy paths) ----

  /**
   * Hash/probe/write/verify for a single float merge vector. Returns the target dict ord. Used by
   * the generic per-doc merge loop for non-dedup sources.
   */
  private int addFloatMergeVector(float[] vec, int mergeFieldIdx, int subIdx, int sourceOrd)
      throws IOException {
    long hash = hashFloats(vec);
    for (int probe = 0; ; probe++) {
      long probeHash = hash + probe;
      int existing = hashToOrd.getOrDefault(probeHash, -1);
      if (existing < 0) {
        int dictOrd = dictSize++;
        hashToOrd.put(probeHash, dictOrd);
        writeFloatsToData(vec);
        recordMergeSource(dictOrd, mergeFieldIdx, subIdx, sourceOrd);
        return dictOrd;
      }
      float[] candidate = (float[]) dictSourceVector(existing);
      if (Arrays.equals(vec, candidate)) {
        return existing;
      }
    }
  }

  /** Byte-encoded mirror of {@link #addFloatMergeVector(float[], int, int, int)}. */
  private int addByteMergeVector(byte[] vec, int mergeFieldIdx, int subIdx, int sourceOrd)
      throws IOException {
    long hash = hashBytes(vec, vec.length);
    for (int probe = 0; ; probe++) {
      long probeHash = hash + probe;
      int existing = hashToOrd.getOrDefault(probeHash, -1);
      if (existing < 0) {
        int dictOrd = dictSize++;
        hashToOrd.put(probeHash, dictOrd);
        data.writeBytes(vec, 0, vec.length);
        recordMergeSource(dictOrd, mergeFieldIdx, subIdx, sourceOrd);
        return dictOrd;
      }
      byte[] candidate = (byte[]) dictSourceVector(existing);
      if (Arrays.equals(vec, candidate)) {
        return existing;
      }
    }
  }

  /**
   * Hash/probe/write/verify for a float vector coming from a {@link DedupAwareFloatSub}. On a hash
   * hit, first inspect whether the candidate was contributed by the <em>same</em> source sub:
   *
   * <ul>
   *   <li>Same sub, same {@code srcDictOrd} → identical vector (source format guarantees distinct
   *       source dict ords correspond to distinct vectors). Return the candidate's target ord
   *       without fetching either vector's bytes.
   *   <li>Same sub, different {@code srcDictOrd} → guaranteed-distinct vector. Continue probing
   *       without fetching either vector's bytes.
   *   <li>Cross-sub (different sub, or flush source) → byte equality via {@link
   *       #dictSourceVector(int)} (correctness is required for cross-segment dedup).
   * </ul>
   */
  private int addDedupAwareFloatVector(float[] vec, int mergeFieldIdx, int subIdx, int srcDictOrd)
      throws IOException {
    long hash = hashFloats(vec);
    long curHigh = packedMergeHigh32(mergeFieldIdx, subIdx);
    for (int probe = 0; ; probe++) {
      long probeHash = hash + probe;
      int existing = hashToOrd.getOrDefault(probeHash, -1);
      if (existing < 0) {
        int dictOrd = dictSize++;
        hashToOrd.put(probeHash, dictOrd);
        writeFloatsToData(vec);
        recordMergeSource(dictOrd, mergeFieldIdx, subIdx, srcDictOrd);
        return dictOrd;
      }
      long candSrc = dictSources[existing];
      if ((candSrc >>> 32) == curHigh) {
        // Same sub — int-ord comparison is authoritative.
        if ((int) candSrc == srcDictOrd) {
          return existing;
        }
        // Different srcDictOrd within same sub → distinct vector. Continue probing.
      } else {
        // Cross-sub / flush — full byte equality.
        float[] candidate = (float[]) dictSourceVector(existing);
        if (Arrays.equals(vec, candidate)) {
          return existing;
        }
      }
    }
  }

  /** Byte-encoded mirror of {@link #addDedupAwareFloatVector(float[], int, int, int)}. */
  private int addDedupAwareByteVector(byte[] vec, int mergeFieldIdx, int subIdx, int srcDictOrd)
      throws IOException {
    long hash = hashBytes(vec, vec.length);
    long curHigh = packedMergeHigh32(mergeFieldIdx, subIdx);
    for (int probe = 0; ; probe++) {
      long probeHash = hash + probe;
      int existing = hashToOrd.getOrDefault(probeHash, -1);
      if (existing < 0) {
        int dictOrd = dictSize++;
        hashToOrd.put(probeHash, dictOrd);
        data.writeBytes(vec, 0, vec.length);
        recordMergeSource(dictOrd, mergeFieldIdx, subIdx, srcDictOrd);
        return dictOrd;
      }
      long candSrc = dictSources[existing];
      if ((candSrc >>> 32) == curHigh) {
        if ((int) candSrc == srcDictOrd) {
          return existing;
        }
      } else {
        byte[] candidate = (byte[]) dictSourceVector(existing);
        if (Arrays.equals(vec, candidate)) {
          return existing;
        }
      }
    }
  }

  /**
   * Return the high 32 bits of the packed source descriptor for a merge-contributed dict ord from
   * {@code (mergeFieldIdx, subIdx)}. Two merge entries share these bits iff they came from the same
   * sub. A flush entry has top bit 0 and so can never match a merge entry here.
   */
  private static long packedMergeHigh32(int mergeFieldIdx, int subIdx) {
    return (1L << 31) | ((long) mergeFieldIdx << 16) | (subIdx & 0xFFFFL);
  }

  // ---- Source lookup + verification helpers ----

  /**
   * Resolve {@code dictOrd}'s source vector as a typed array ({@code float[]} or {@code byte[]}).
   * For flush-contributed ords, this is the on-heap {@link FieldWriter#vectors} entry. For merge-
   * contributed ords, this is a random-access fetch through the sub's dedicated {@code
   * verifyValues} reader (separate from the iteration reader, so there is no aliasing with the
   * iterator's current buffer).
   */
  private Object dictSourceVector(int dictOrd) throws IOException {
    long src = dictSources[dictOrd];
    if ((src >>> 63) == 0) {
      int candidateFieldIdx = (int) (src >>> 32);
      int candidateInFieldOrd = (int) src;
      return fields.get(candidateFieldIdx).vectors.get(candidateInFieldOrd);
    } else {
      int mergeFieldIdx = (int) ((src >>> 48) & 0x7FFFL);
      int subIdx = (int) ((src >>> 32) & 0xFFFFL);
      int sourceOrd = (int) src;
      return mergeFieldSubs.get(mergeFieldIdx).get(subIdx).valueAt(sourceOrd);
    }
  }

  /** Typed equality between two same-encoding vector arrays. */
  private boolean typedVectorsEqual(Object a, Object b) {
    if (a instanceof float[] af) {
      return Arrays.equals(af, (float[]) b);
    } else if (a instanceof byte[] ab) {
      return Arrays.equals(ab, (byte[]) b);
    } else {
      throw new AssertionError("Unexpected vector type: " + a.getClass());
    }
  }

  /** Write a vector directly to {@link #data} without an intermediate {@code byte[]}. */
  private void writeVectorToData(Object vec) throws IOException {
    if (vec instanceof float[] f) {
      writeFloatsToData(f);
    } else if (vec instanceof byte[] b) {
      data.writeBytes(b, 0, b.length);
    } else {
      throw new AssertionError("Unexpected vector type: " + vec.getClass());
    }
  }

  /**
   * Write a float vector to {@link #data} via the reusable {@link #floatWriteBuffer}. This mirrors
   * {@code Lucene99FlatVectorsWriter.writeFloat32Vectors}: each call rewrites the buffer in place
   * and issues a single bulk {@link IndexOutput#writeBytes(byte[], int, int)}. Faster than a
   * per-element {@code data.writeInt(Float.floatToRawIntBits(f))} loop, particularly on buffered
   * outputs where each virtual call carries non-trivial overhead.
   */
  private void writeFloatsToData(float[] vec) throws IOException {
    floatWriteBuffer.asFloatBuffer().put(vec);
    data.writeBytes(floatWriteBuffer.array(), floatWriteBuffer.array().length);
  }

  // ---- Shared dict state + source tracking ----

  private void initDict(FieldInfo fieldInfo) {
    int dim = fieldInfo.getVectorDimension();
    VectorEncoding enc = fieldInfo.getVectorEncoding();

    if (dictInitialized && dim == dictDimension && enc == dictEncoding) {
      return;
    }

    hashToOrd.clear();
    dictSources = null;
    mergeFieldSubs.clear();
    dictSize = 0;
    dictDataOffset = data.getFilePointer();
    dictDimension = dim;
    dictEncoding = enc;
    // Allocate a reusable little-endian byte buffer for float writes (see writeFloatsToData).
    // Null for byte-encoded groups since those write through data.writeBytes directly.
    floatWriteBuffer =
        (enc == VectorEncoding.FLOAT32)
            ? ByteBuffer.allocate(dim * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN)
            : null;
    dictInitialized = true;
  }

  /** Record a flush-contributed source in {@link #dictSources} (top bit 0). */
  private void recordFlushSource(int dictOrd, int fieldIdx, int inFieldOrd) {
    ensureDictSourcesCapacity(dictOrd);
    dictSources[dictOrd] = ((long) fieldIdx << 32) | (inFieldOrd & 0xFFFFFFFFL);
  }

  /** Record a merge-contributed source in {@link #dictSources} (top bit 1). */
  private void recordMergeSource(int dictOrd, int mergeFieldIdx, int subIdx, int sourceOrd) {
    ensureDictSourcesCapacity(dictOrd);
    if (mergeFieldIdx >= (1 << 15) || subIdx >= (1 << 16)) {
      throw new IllegalStateException(
          "Too many merge fields or source segments: mergeFieldIdx="
              + mergeFieldIdx
              + ", subIdx="
              + subIdx);
    }
    dictSources[dictOrd] =
        (1L << 63)
            | ((long) mergeFieldIdx << 48)
            | ((long) subIdx << 32)
            | (sourceOrd & 0xFFFFFFFFL);
  }

  private void ensureDictSourcesCapacity(int dictOrd) {
    if (dictSources == null || dictOrd >= dictSources.length) {
      int newLen = dictSources == null ? 16 : ArrayUtil.oversize(dictOrd + 1, Long.BYTES);
      long[] grown = new long[newLen];
      if (dictSources != null) {
        System.arraycopy(dictSources, 0, grown, 0, dictSources.length);
      }
      dictSources = grown;
    }
  }

  /**
   * Build one {@link DedupSub} per source segment that has float values for this field, populating
   * the passed-in {@code subs} list. For sources that are {@link DedupFlatVectorsReader}s with
   * {@code ordToDict != null}, construct a {@link DedupAwareFloatSub} with a lazy cache keyed on
   * source dict ord. No precompute — each source dict ord is hashed on-demand the first time a live
   * doc references it, so dead vectors in the source's dictionary are naturally dropped.
   */
  private void buildFloatSubs(List<DedupSub<?>> subs, FieldInfo fieldInfo, MergeState mergeState)
      throws IOException {
    int subIdx = 0;
    for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
      if (!KnnVectorsWriter.MergedVectorValues.hasVectorValues(
          mergeState.fieldInfos[i], fieldInfo.name)) {
        continue;
      }
      var reader = mergeState.knnVectorsReaders[i];
      if (reader == null) continue;
      FloatVectorValues values = reader.getFloatVectorValues(fieldInfo.name);
      if (values == null) continue;

      if (reader instanceof DedupFlatVectorsReader dedupReader) {
        int[] sourceOrdToDict = dedupReader.getOrdToDict(fieldInfo.name);
        if (sourceOrdToDict != null) {
          FloatVectorValues dictVerifyValues = dedupReader.getDictFloatVectorValues(fieldInfo.name);
          subs.add(
              new DedupAwareFloatSub(
                  subIdx++,
                  mergeState.docMaps[i],
                  values,
                  dictVerifyValues,
                  sourceOrdToDict,
                  dictVerifyValues.size()));
          continue;
        }
      }

      // Generic path: separate verify handle keeps random-access verification reads from aliasing
      // the iteration buffer within the same sub.
      FloatVectorValues verifyValues = values.copy();
      subs.add(new FloatDedupSub(subIdx++, mergeState.docMaps[i], values, verifyValues));
    }
  }

  /** Byte-encoded mirror of {@link #buildFloatSubs}. */
  private void buildByteSubs(List<DedupSub<?>> subs, FieldInfo fieldInfo, MergeState mergeState)
      throws IOException {
    int subIdx = 0;
    for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
      if (!KnnVectorsWriter.MergedVectorValues.hasVectorValues(
          mergeState.fieldInfos[i], fieldInfo.name)) {
        continue;
      }
      var reader = mergeState.knnVectorsReaders[i];
      if (reader == null) continue;
      ByteVectorValues values = reader.getByteVectorValues(fieldInfo.name);
      if (values == null) continue;

      if (reader instanceof DedupFlatVectorsReader dedupReader) {
        int[] sourceOrdToDict = dedupReader.getOrdToDict(fieldInfo.name);
        if (sourceOrdToDict != null) {
          ByteVectorValues dictVerifyValues = dedupReader.getDictByteVectorValues(fieldInfo.name);
          subs.add(
              new DedupAwareByteSub(
                  subIdx++,
                  mergeState.docMaps[i],
                  values,
                  dictVerifyValues,
                  sourceOrdToDict,
                  dictVerifyValues.size()));
          continue;
        }
      }

      ByteVectorValues verifyValues = values.copy();
      subs.add(new ByteDedupSub(subIdx++, mergeState.docMaps[i], values, verifyValues));
    }
  }

  /**
   * 64-bit hash over a float vector. Operates directly on {@link Float#floatToRawIntBits(float)}
   * with no byte materialization. The on-disk byte layout of each float (little-endian 4-byte int
   * via {@link IndexOutput#writeInt(int)}) yields identical hash input to a byte-oriented FNV
   * computation, so we produce the same high-quality 64-bit digest without touching bytes.
   */
  static long hashFloats(float[] vec) {
    int h1 = 1;
    int h2 = 0x811c9dc5;
    for (float f : vec) {
      int bits = Float.floatToRawIntBits(f);
      h1 = 31 * h1 + bits;
      // FNV-1a on the 4 little-endian bytes of `bits`.
      h2 = (h2 ^ (bits & 0xff)) * 0x01000193;
      h2 = (h2 ^ ((bits >>> 8) & 0xff)) * 0x01000193;
      h2 = (h2 ^ ((bits >>> 16) & 0xff)) * 0x01000193;
      h2 = (h2 ^ ((bits >>> 24) & 0xff)) * 0x01000193;
    }
    return ((long) h1 << 32) | (h2 & 0xFFFFFFFFL);
  }

  /**
   * 64-bit hash over a byte vector. Mirror of {@link #hashFloats(float[])} for byte-encoded
   * vectors. Since a single writer instance processes one encoding per shared-dict group, the two
   * hash functions don't need to agree on the same input.
   */
  static long hashBytes(byte[] bytes, int length) {
    int h1 = 1;
    int h2 = 0x811c9dc5;
    for (int i = 0; i < length; i++) {
      int b = bytes[i] & 0xFF;
      h2 = (h2 ^ b) * 0x01000193;
      if ((i & 3) == 0) {
        h1 = 31 * h1;
      }
      h1 += b << ((i & 3) * 8);
    }
    return ((long) h1 << 32) | (h2 & 0xFFFFFFFFL);
  }

  // ---- Finish / close ----

  @Override
  public void finish() throws IOException {
    if (finished) throw new IllegalStateException("already finished");
    finished = true;

    // All dictionary bytes are already in `data`. Release cached merge subs.
    mergeFieldSubs.clear();

    long[] mapOffsets = new long[pendingFields.size()];
    long[] mapLengths = new long[pendingFields.size()];
    for (int i = 0; i < pendingFields.size(); i++) {
      PendingField pf = pendingFields.get(i);
      mapOffsets[i] = data.getFilePointer();
      if (pf.ordToDict != null) {
        for (int ord = 0; ord < pf.size; ord++) {
          data.writeVInt(pf.ordToDict[ord]);
        }
      }
      mapLengths[i] = data.getFilePointer() - mapOffsets[i];
    }

    for (int i = 0; i < pendingFields.size(); i++) {
      PendingField pf = pendingFields.get(i);
      meta.writeInt(pf.fieldInfo.number);
      meta.writeVInt(pf.size);
      meta.writeInt(pf.fieldInfo.getVectorSimilarityFunction().ordinal());
      meta.writeVInt(pf.dictSize);
      meta.writeVInt(pf.dictDimension);
      meta.writeVInt(pf.dictEncoding != null ? pf.dictEncoding.ordinal() : -1);
      meta.writeVLong(pf.dictDataOffset);
      meta.writeVLong(mapOffsets[i]);
      meta.writeVLong(mapLengths[i]);
      OrdToDocDISIReaderConfiguration.writeStoredMeta(
          DIRECT_MONOTONIC_BLOCK_SHIFT, meta, data, pf.size, pf.maxDoc, pf.docsWithField);
    }

    meta.writeInt(-1); // end-of-fields sentinel
    CodecUtil.writeFooter(meta);
    CodecUtil.writeFooter(data);
  }

  @Override
  public long ramBytesUsed() {
    long total = RamUsageEstimator.shallowSizeOfInstance(DedupFlatVectorsWriter.class);
    for (FieldWriter<?> field : fields) {
      total += field.ramBytesUsed();
    }
    for (PendingField pf : pendingFields) {
      total += pf.docsWithField.ramBytesUsed();
      if (pf.ordToDict != null) {
        total += (long) pf.ordToDict.length * Integer.BYTES;
      }
    }
    total += hashToOrd.size() * 16L;
    if (dictSources != null) {
      total += (long) dictSources.length * Long.BYTES;
    }
    return total;
  }

  @Override
  public void close() throws IOException {
    mergeFieldSubs.clear();
    IOUtils.close(meta, data);
  }

  // ---- Inner types ----

  /**
   * Per-field metadata accumulated during flush/merge, written to disk in {@link #finish()}. The
   * {@code ordToDict} array is null for empty fields.
   */
  private record PendingField(
      FieldInfo fieldInfo,
      int size,
      int maxDoc,
      DocsWithFieldSet docsWithField,
      int[] ordToDict,
      int dictSize,
      long dictDataOffset,
      int dictDimension,
      VectorEncoding dictEncoding) {}

  /**
   * One sub-reader for the merge path, wrapping a per-segment vector values handle plus a separate
   * {@code verifyValues} handle used exclusively for random-access collision verification reads.
   * The two handles share the underlying file but maintain their own buffers, so verification
   * against a candidate vector from the same sub never corrupts the iteration buffer.
   */
  private abstract static class DedupSub<V> extends DocIDMerger.Sub {
    final int subIdx;
    final KnnVectorValues.DocIndexIterator iterator;

    DedupSub(int subIdx, MergeState.DocMap docMap, KnnVectorValues values) {
      super(docMap);
      this.subIdx = subIdx;
      this.iterator = values.iterator();
      assert iterator.docID() == -1;
    }

    @Override
    public int nextDoc() throws IOException {
      return iterator.nextDoc();
    }

    int iteratorIndex() {
      return iterator.index();
    }

    /**
     * Random-access fetch through the dedicated verification handle; never returns a buffer aliased
     * with the iteration buffer.
     */
    abstract V valueAt(int sourceOrd) throws IOException;
  }

  private static final class FloatDedupSub extends DedupSub<float[]> {
    private final FloatVectorValues values;
    private final FloatVectorValues verifyValues;

    FloatDedupSub(
        int subIdx,
        MergeState.DocMap docMap,
        FloatVectorValues values,
        FloatVectorValues verifyValues) {
      super(subIdx, docMap, values);
      this.values = values;
      this.verifyValues = verifyValues;
    }

    float[] current() throws IOException {
      return values.vectorValue(iterator.index());
    }

    @Override
    float[] valueAt(int sourceOrd) throws IOException {
      return verifyValues.vectorValue(sourceOrd);
    }
  }

  /**
   * Specialized float sub for sources that are themselves {@link DedupFlatVectorsReader}s. Carries
   * the source's {@code ordToDict} mapping, a lazy per-source-dict-ord cache of resolved target
   * dict ords, and a separate dict-level verify handle used when a different merge sub has a hash
   * collision with a vector this sub previously contributed.
   *
   * <p>The cache is populated on-demand during per-doc iteration: each unique source dict ord is
   * hashed at most once regardless of how many live docs reference it. Source dict ords that are
   * never referenced by any live doc are never hashed, so dead vectors in the source's dictionary
   * are naturally dropped during merge.
   *
   * <p>Within the same sub, vector-equality on hash collision is decided purely by comparing {@link
   * #sourceOrdToDict}{@code [currentDocOrd]} against the candidate's recorded source dict ord — the
   * source format guarantees distinct source dict ords correspond to distinct vectors, so
   * byte-level comparison is unnecessary. Cross-sub collisions still fall back to byte equality,
   * using {@link #dictVerifyValues} (backed by the source's dictionary slice) to read the candidate
   * vector by its source dict ord.
   */
  private static final class DedupAwareFloatSub extends DedupSub<float[]> {
    final int[] sourceOrdToDict;
    final int[] sourceDictOrdCache;
    private final FloatVectorValues iterValues;
    private final FloatVectorValues dictVerifyValues;

    DedupAwareFloatSub(
        int subIdx,
        MergeState.DocMap docMap,
        FloatVectorValues iterValues,
        FloatVectorValues dictVerifyValues,
        int[] sourceOrdToDict,
        int sourceDictSize) {
      super(subIdx, docMap, iterValues);
      this.iterValues = iterValues;
      this.dictVerifyValues = dictVerifyValues;
      this.sourceOrdToDict = sourceOrdToDict;
      this.sourceDictOrdCache = new int[sourceDictSize];
      Arrays.fill(this.sourceDictOrdCache, -1);
    }

    float[] current() throws IOException {
      return iterValues.vectorValue(iterator.index());
    }

    @Override
    float[] valueAt(int sourceDictOrd) throws IOException {
      return dictVerifyValues.vectorValue(sourceDictOrd);
    }
  }

  private static final class ByteDedupSub extends DedupSub<byte[]> {
    private final ByteVectorValues values;
    private final ByteVectorValues verifyValues;

    ByteDedupSub(
        int subIdx,
        MergeState.DocMap docMap,
        ByteVectorValues values,
        ByteVectorValues verifyValues) {
      super(subIdx, docMap, values);
      this.values = values;
      this.verifyValues = verifyValues;
    }

    byte[] current() throws IOException {
      return values.vectorValue(iterator.index());
    }

    @Override
    byte[] valueAt(int sourceOrd) throws IOException {
      return verifyValues.vectorValue(sourceOrd);
    }
  }

  /** Byte-encoded mirror of {@link DedupAwareFloatSub}. */
  private static final class DedupAwareByteSub extends DedupSub<byte[]> {
    final int[] sourceOrdToDict;
    final int[] sourceDictOrdCache;
    private final ByteVectorValues iterValues;
    private final ByteVectorValues dictVerifyValues;

    DedupAwareByteSub(
        int subIdx,
        MergeState.DocMap docMap,
        ByteVectorValues iterValues,
        ByteVectorValues dictVerifyValues,
        int[] sourceOrdToDict,
        int sourceDictSize) {
      super(subIdx, docMap, iterValues);
      this.iterValues = iterValues;
      this.dictVerifyValues = dictVerifyValues;
      this.sourceOrdToDict = sourceOrdToDict;
      this.sourceDictOrdCache = new int[sourceDictSize];
      Arrays.fill(this.sourceDictOrdCache, -1);
    }

    byte[] current() throws IOException {
      return iterValues.vectorValue(iterator.index());
    }

    @Override
    byte[] valueAt(int sourceDictOrd) throws IOException {
      return dictVerifyValues.vectorValue(sourceDictOrd);
    }
  }

  /** Buffers vectors in memory during indexing (flush path only). */
  private abstract static class FieldWriter<T> extends FlatFieldVectorsWriter<T> {
    final FieldInfo fieldInfo;
    final int dim;
    final DocsWithFieldSet docsWithField;
    final List<T> vectors;
    private boolean finished;
    private int lastDocID = -1;

    static FieldWriter<?> create(FieldInfo fieldInfo) {
      return switch (fieldInfo.getVectorEncoding()) {
        case BYTE ->
            new FieldWriter<byte[]>(fieldInfo) {
              @Override
              public byte[] copyValue(byte[] v) {
                return ArrayUtil.copyOfSubArray(v, 0, dim);
              }
            };
        case FLOAT32 ->
            new FieldWriter<float[]>(fieldInfo) {
              @Override
              public float[] copyValue(float[] v) {
                return ArrayUtil.copyOfSubArray(v, 0, dim);
              }
            };
      };
    }

    FieldWriter(FieldInfo fieldInfo) {
      this.fieldInfo = fieldInfo;
      this.dim = fieldInfo.getVectorDimension();
      this.docsWithField = new DocsWithFieldSet();
      this.vectors = new ArrayList<>();
    }

    @Override
    public void addValue(int docID, T vectorValue) throws IOException {
      if (finished) throw new IllegalStateException("already finished");
      if (docID == lastDocID) {
        throw new IllegalArgumentException(
            "VectorValuesField \"" + fieldInfo.name + "\" appears more than once in this document");
      }
      assert docID > lastDocID;
      docsWithField.add(docID);
      vectors.add(copyValue(vectorValue));
      lastDocID = docID;
    }

    @Override
    public long ramBytesUsed() {
      if (vectors.isEmpty()) return 0;
      return docsWithField.ramBytesUsed()
          + (long) vectors.size()
              * (RamUsageEstimator.NUM_BYTES_OBJECT_REF + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER)
          + (long) vectors.size() * dim * fieldInfo.getVectorEncoding().byteSize;
    }

    @Override
    public List<T> getVectors() {
      return vectors;
    }

    @Override
    public DocsWithFieldSet getDocsWithFieldSet() {
      return docsWithField;
    }

    @Override
    public void finish() {
      finished = true;
    }

    @Override
    public boolean isFinished() {
      return finished;
    }
  }
}
