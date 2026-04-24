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

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

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
import org.apache.lucene.codecs.lucene95.OffHeapByteVectorValues;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
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
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;

/**
 * Writer that de-duplicates vectors across all fields in a segment. Unique vectors are stored in a
 * contiguous dictionary region in the .dvd file. Each field stores a compact ordinal mapping from
 * document ords to dictionary ords.
 *
 * <p>Merge uses a two-pass approach per field. Pass 1 writes all vectors (including duplicates) to
 * a temp file. Pass 2 reads back from the closed temp file, hashes, deduplicates against the
 * dictionary, and builds the ordToDict mapping. Only the hash map (~16 bytes/unique vector) and the
 * ordinal mapping (4 bytes/doc) are held in memory.
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
   * Dictionary vectors are written to a temp file, then bulk-copied to {@code data} at finish time.
   * During flush, this file is also read back for collision verification (with mode toggling).
   * During merge, collision verification reads from the per-field temp file instead.
   */
  private IndexOutput dictTempOut;

  private String dictTempName;

  /** Read handle for collision verification. Opened lazily, closed on next write. */
  private IndexInput dictReadHandle;

  private int dictSize;
  private long dictDataOffset;
  private int dictDimension;
  private VectorEncoding dictEncoding;
  private int vectorByteSize;

  /** Reusable byte buffer for serializing float vectors. Used for both hashing and writing. */
  private byte[] vectorBytes;

  /** Reusable byte buffer for bulk collision verification reads. */
  private byte[] compareBytes;

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
    for (FieldWriter<?> field : fields) {
      flushField(field, maxDoc, sortMap);
      field.finish();
    }
  }

  @SuppressWarnings("unchecked")
  private <T> void flushField(FieldWriter<T> field, int maxDoc, Sorter.DocMap sortMap)
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

    // Apply sort map if needed
    int[] ordMap = null;
    if (sortMap != null) {
      ordMap = new int[size];
      DocsWithFieldSet newDocsWithField = new DocsWithFieldSet();
      KnnVectorsWriter.mapOldOrdToNewOrd(docsWithField, sortMap, null, ordMap, newDocsWithField);
      docsWithField = newDocsWithField;
    }

    // Dedup and build mapping in memory (flush buffers are bounded by indexing buffer size)
    int[] ordToDict = new int[size];
    for (int newOrd = 0; newOrd < size; newOrd++) {
      int srcOrd = ordMap != null ? ordMap[newOrd] : newOrd;
      ordToDict[newOrd] = addToDict(vectors.get(srcOrd));
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

  // ---- Merge path (two-pass) ----

  /**
   * Pass 1: iterate merged vector values and write all vectors (including duplicates) to a temp
   * file. Returns the number of vectors written and populates {@code docsWithField}.
   */
  private int writeMergedVectorsToTemp(
      FieldInfo fieldInfo,
      MergeState mergeState,
      IndexOutput tempOut,
      DocsWithFieldSet docsWithField)
      throws IOException {
    VectorEncoding encoding = fieldInfo.getVectorEncoding();
    int byteSize = fieldInfo.getVectorDimension() * encoding.byteSize;
    KnnVectorValues merged =
        switch (encoding) {
          case FLOAT32 ->
              KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
          case BYTE ->
              KnnVectorsWriter.MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState);
        };

    int count = 0;
    KnnVectorValues.DocIndexIterator iter = merged.iterator();
    for (int doc = iter.nextDoc(); doc != NO_MORE_DOCS; doc = iter.nextDoc()) {
      byte[] rawBytes = vectorToBytes(merged, iter.index());
      tempOut.writeBytes(rawBytes, 0, byteSize);
      docsWithField.add(doc);
      count++;
    }
    return count;
  }

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    initDict(fieldInfo);
    int maxDoc = segmentWriteState.segmentInfo.maxDoc();
    int byteSize = fieldInfo.getVectorDimension() * fieldInfo.getVectorEncoding().byteSize;

    // Pass 1: write all vectors to a temp file.
    IndexOutput tempOut =
        segmentWriteState.directory.createTempOutput(
            data.getName(), "dedup-merge", segmentWriteState.context);
    String tempName = tempOut.getName();
    try {
      DocsWithFieldSet docsWithField = new DocsWithFieldSet();
      int count = writeMergedVectorsToTemp(fieldInfo, mergeState, tempOut, docsWithField);
      CodecUtil.writeFooter(tempOut);
      IOUtils.close(tempOut);

      if (count == 0) {
        pendingFields.add(
            new PendingField(
                fieldInfo,
                0,
                maxDoc,
                docsWithField,
                null,
                dictSize,
                dictDataOffset,
                dictDimension,
                dictEncoding));
        return;
      }

      // Pass 2: dedup from the closed temp file.
      try (IndexInput tempIn =
          segmentWriteState.directory.openInput(tempName, segmentWriteState.context)) {
        dedupFromTempFile(fieldInfo, tempIn, count, maxDoc, docsWithField, byteSize);
      }
    } finally {
      IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, tempName);
    }
  }

  @Override
  public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(
      FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    initDict(fieldInfo);
    int maxDoc = segmentWriteState.segmentInfo.maxDoc();
    VectorEncoding encoding = fieldInfo.getVectorEncoding();
    int byteSize = fieldInfo.getVectorDimension() * encoding.byteSize;

    // Pass 1: write all vectors to a temp file (also used as the HNSW scorer source).
    IndexOutput scorerTempOut =
        segmentWriteState.directory.createTempOutput(
            data.getName(), "dedup-scorer", segmentWriteState.context);
    IndexInput scorerTempIn = null;
    try {
      DocsWithFieldSet docsWithField = new DocsWithFieldSet();
      int count = writeMergedVectorsToTemp(fieldInfo, mergeState, scorerTempOut, docsWithField);
      CodecUtil.writeFooter(scorerTempOut);
      IOUtils.close(scorerTempOut);

      // Pass 2: dedup from the closed temp file.
      if (count > 0) {
        try (IndexInput tempIn =
            segmentWriteState.directory.openInput(
                scorerTempOut.getName(), segmentWriteState.context)) {
          dedupFromTempFile(fieldInfo, tempIn, count, maxDoc, docsWithField, byteSize);
        }
      } else {
        pendingFields.add(
            new PendingField(
                fieldInfo,
                0,
                maxDoc,
                docsWithField,
                null,
                dictSize,
                dictDataOffset,
                dictDimension,
                dictEncoding));
      }

      // Reopen with random-access hints for HNSW scorer construction.
      scorerTempIn =
          segmentWriteState.directory.openInput(
              scorerTempOut.getName(),
              IOContext.DEFAULT.withHints(
                  FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM));

      KnnVectorValues scorerValues =
          switch (encoding) {
            case FLOAT32 ->
                new OffHeapFloatVectorValues.DenseOffHeapVectorValues(
                    fieldInfo.getVectorDimension(),
                    count,
                    scorerTempIn,
                    byteSize,
                    vectorsScorer,
                    fieldInfo.getVectorSimilarityFunction());
            case BYTE ->
                new OffHeapByteVectorValues.DenseOffHeapVectorValues(
                    fieldInfo.getVectorDimension(),
                    count,
                    scorerTempIn,
                    byteSize,
                    vectorsScorer,
                    fieldInfo.getVectorSimilarityFunction());
          };

      RandomVectorScorerSupplier supplier =
          vectorsScorer.getRandomVectorScorerSupplier(
              fieldInfo.getVectorSimilarityFunction(), scorerValues);

      final int finalCount = count;
      final IndexInput finalTempIn = scorerTempIn;
      final String tempFileName = scorerTempOut.getName();
      scorerTempIn = null; // ownership transferred
      return new CloseableRandomVectorScorerSupplier() {
        @Override
        public UpdateableRandomVectorScorer scorer() throws IOException {
          return supplier.scorer();
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
          return supplier.copy();
        }

        @Override
        public int totalVectorCount() {
          return finalCount;
        }

        @Override
        public void close() throws IOException {
          IOUtils.close(finalTempIn);
          IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, tempFileName);
        }
      };
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, scorerTempIn, scorerTempOut);
      IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, scorerTempOut.getName());
      throw t;
    }
  }

  /**
   * Pass 2 of the two-pass merge. Reads vectors from a closed temp file, hashes and deduplicates
   * them against the dictionary, and builds the ordToDict mapping.
   *
   * <p>Phase A (dict in read mode): scans all vectors, resolves each to a dictOrd via hash lookup
   * and byte-for-byte verification. New unique vectors are assigned dictOrds but not yet written.
   *
   * <p>Phase B (dict in write mode): seeks to each new unique vector in the temp file and appends
   * it to the dictionary. Only new vectors are read — duplicates are skipped.
   *
   * <p>The dict temp file is toggled once per field (read for phase A, write for phase B), not per
   * vector.
   */
  private void dedupFromTempFile(
      FieldInfo fieldInfo,
      IndexInput tempIn,
      int count,
      int maxDoc,
      DocsWithFieldSet docsWithField,
      int byteSize)
      throws IOException {
    int[] ordToDict = new int[count];
    int dictSizeBefore = dictSize;

    // Phase A: dict in read mode — hash, dedup, build ordToDict.
    IndexInput dictIn = getDictReadHandle();
    // Temp-file indices of new unique vectors (for phase B writes). Sized to count as upper bound.
    int numNew = 0;
    int[] newVectorTempIndices = new int[count];

    for (int i = 0; i < count; i++) {
      tempIn.seek((long) i * byteSize);
      tempIn.readBytes(vectorBytes, 0, byteSize);
      long hash = hashBytes(vectorBytes, byteSize);

      int dictOrd = -1;
      for (int probe = 0; ; probe++) {
        long probeHash = hash + probe;
        int existing = hashToOrd.getOrDefault(probeHash, -1);
        if (existing < 0) {
          // New unique vector
          dictOrd = dictSize++;
          hashToOrd.put(probeHash, dictOrd);
          newVectorTempIndices[numNew++] = i;
          break;
        }
        // Hash hit — verify byte-for-byte equality
        IndexInput verifySource;
        if (existing >= dictSizeBefore) {
          // Candidate was added by this field — read from temp file
          tempIn.seek((long) newVectorTempIndices[existing - dictSizeBefore] * byteSize);
          verifySource = tempIn;
        } else {
          // Candidate from a previous field — read from dict temp file
          dictIn.seek((long) existing * byteSize);
          verifySource = dictIn;
        }
        verifySource.readBytes(compareBytes, 0, byteSize);
        if (Arrays.equals(vectorBytes, 0, byteSize, compareBytes, 0, byteSize)) {
          dictOrd = existing;
          break;
        }
        // Hash collision — continue probing
      }
      ordToDict[i] = dictOrd;
    }

    // Phase B: dict in write mode — append new unique vectors.
    ensureDictWritable();
    for (int n = 0; n < numNew; n++) {
      tempIn.seek((long) newVectorTempIndices[n] * byteSize);
      tempIn.readBytes(vectorBytes, 0, byteSize);
      dictTempOut.writeBytes(vectorBytes, 0, byteSize);
    }

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

  /**
   * Serialize a vector to its raw byte representation. Reuses the {@link #vectorBytes} buffer. The
   * returned array is {@link #vectorBytes} — callers must consume it before the next call.
   */
  private byte[] vectorToBytes(KnnVectorValues values, int index) throws IOException {
    if (values instanceof FloatVectorValues fv) {
      float[] vec = fv.vectorValue(index);
      ByteBuffer buf = ByteBuffer.wrap(vectorBytes).order(ByteOrder.LITTLE_ENDIAN);
      buf.asFloatBuffer().put(vec);
    } else {
      byte[] vec = ((ByteVectorValues) values).vectorValue(index);
      System.arraycopy(vec, 0, vectorBytes, 0, vectorByteSize);
    }
    return vectorBytes;
  }

  // ---- Dictionary management ----

  private void initDict(FieldInfo fieldInfo) throws IOException {
    int dim = fieldInfo.getVectorDimension();
    VectorEncoding enc = fieldInfo.getVectorEncoding();

    // Reuse existing dictionary for cross-field dedup when dim/encoding match
    if (dictTempName != null && dim == dictDimension && enc == dictEncoding) {
      ensureDictWritable();
      return;
    }

    // Different dim/encoding or first field — start a new dictionary
    if (dictTempName != null) {
      copyDictTempToData();
    }
    IOUtils.close(dictReadHandle);
    dictReadHandle = null;
    hashToOrd.clear();
    dictSize = 0;
    dictDataOffset = data.getFilePointer();
    dictDimension = dim;
    dictEncoding = enc;
    vectorByteSize = dim * enc.byteSize;
    vectorBytes = new byte[vectorByteSize];
    compareBytes = new byte[vectorByteSize];
    dictTempOut =
        segmentWriteState.directory.createTempOutput(
            data.getName(), "dedup-dict", segmentWriteState.context);
    dictTempName = dictTempOut.getName();
  }

  /**
   * Add a vector (already in its in-memory representation) to the dictionary. Used by the flush
   * path where vectors are in typed arrays.
   */
  private <T> int addToDict(T vec) throws IOException {
    // Serialize to raw bytes for hashing and writing
    if (vec instanceof float[] f) {
      ByteBuffer buf = ByteBuffer.wrap(vectorBytes).order(ByteOrder.LITTLE_ENDIAN);
      buf.asFloatBuffer().put(f);
    } else if (vec instanceof byte[] b) {
      System.arraycopy(b, 0, vectorBytes, 0, vectorByteSize);
    }
    return addToDictFromBytes(vectorBytes);
  }

  /**
   * Add a vector (as raw bytes) to the dictionary. Used by the flush path only. Collision
   * verification reads from the dict temp file, which requires toggling between read and write
   * modes. This is acceptable during flush since the number of vectors is bounded by the indexing
   * buffer size.
   */
  private int addToDictFromBytes(byte[] rawBytes) throws IOException {
    long hash = hashBytes(rawBytes, vectorByteSize);

    for (int probe = 0; ; probe++) {
      long probeHash = hash + probe;
      int existing = hashToOrd.getOrDefault(probeHash, -1);
      if (existing < 0) {
        ensureDictWritable();
        int dictOrd = dictSize++;
        hashToOrd.put(probeHash, dictOrd);
        dictTempOut.writeBytes(rawBytes, 0, vectorByteSize);
        return dictOrd;
      }
      // Hash hit — verify equality by reading from dict temp file
      IndexInput reader = getDictReadHandle();
      reader.seek((long) existing * vectorByteSize);
      reader.readBytes(compareBytes, 0, vectorByteSize);
      if (Arrays.equals(rawBytes, 0, vectorByteSize, compareBytes, 0, vectorByteSize)) {
        return existing;
      }
      // Collision — continue probing
    }
  }

  /**
   * Single-pass 64-bit hash over raw bytes. Combines two independent 32-bit hashes (polynomial and
   * FNV-1a) computed in one loop, avoiding the need to iterate the data twice.
   */
  static long hashBytes(byte[] bytes, int length) {
    // Polynomial hash (same algorithm as Arrays.hashCode but on raw bytes in 4-byte groups)
    int h1 = 1;
    // FNV-1a hash
    int h2 = 0x811c9dc5;

    for (int i = 0; i < length; i++) {
      int b = bytes[i] & 0xFF;
      // FNV-1a step
      h2 = (h2 ^ b) * 0x01000193;
      // Accumulate into polynomial hash in 4-byte groups (matching Float.floatToIntBits layout)
      if ((i & 3) == 0) {
        h1 = 31 * h1;
      }
      h1 += b << ((i & 3) * 8);
    }

    return ((long) h1 << 32) | (h2 & 0xFFFFFFFFL);
  }

  // ---- Dictionary temp file management ----

  /** Ensure the dictionary temp file is open for writing. */
  private void ensureDictWritable() throws IOException {
    if (dictReadHandle != null) {
      IOUtils.close(dictReadHandle);
      dictReadHandle = null;
      // Copy existing content to a new temp file to resume writing
      String oldName = dictTempName;
      dictTempOut =
          segmentWriteState.directory.createTempOutput(
              data.getName(), "dedup-dict", segmentWriteState.context);
      dictTempName = dictTempOut.getName();
      try (IndexInput in =
          segmentWriteState.directory.openInput(oldName, segmentWriteState.context)) {
        dictTempOut.copyBytes(in, in.length() - CodecUtil.footerLength());
      }
      IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, oldName);
    }
  }

  /** Open the dictionary temp file for reading (closes the writer first). */
  private IndexInput getDictReadHandle() throws IOException {
    if (dictReadHandle == null) {
      CodecUtil.writeFooter(dictTempOut);
      IOUtils.close(dictTempOut);
      dictTempOut = null;
      dictReadHandle =
          segmentWriteState.directory.openInput(dictTempName, segmentWriteState.context);
    }
    return dictReadHandle;
  }

  /** Copy dictionary vectors from temp file to the main data file. */
  private void copyDictTempToData() throws IOException {
    if (dictTempName == null) return;
    if (dictReadHandle != null) {
      IOUtils.close(dictReadHandle);
      dictReadHandle = null;
    } else if (dictTempOut != null) {
      CodecUtil.writeFooter(dictTempOut);
      IOUtils.close(dictTempOut);
      dictTempOut = null;
    }
    try (IndexInput in =
        segmentWriteState.directory.openInput(dictTempName, segmentWriteState.context)) {
      data.copyBytes(in, in.length() - CodecUtil.footerLength());
    }
    IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, dictTempName);
    dictTempName = null;
  }

  // ---- Finish / close ----

  @Override
  public void finish() throws IOException {
    if (finished) throw new IllegalStateException("already finished");
    finished = true;

    // Copy last field's dictionary from temp to data
    if (dictTempName != null) {
      copyDictTempToData();
    }
    IOUtils.close(dictReadHandle);
    dictReadHandle = null;

    // Write per-field ordToDict mappings to data
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

    // Write per-field metadata to meta, OrdToDoc DISI data to data
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
    return total;
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, data, dictReadHandle, dictTempOut);
    if (dictTempName != null) {
      IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, dictTempName);
    }
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
