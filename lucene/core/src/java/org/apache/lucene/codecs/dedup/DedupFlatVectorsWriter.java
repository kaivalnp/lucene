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
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;

/**
 * Writer that de-duplicates vectors across all fields in a segment.
 *
 * <p>.dvd layout: [header][dict vectors][per-field ordToDict vints][per-field OrdToDoc
 * DISI][footer]
 *
 * <p>At merge time, only a {@code long→int} hash map is in memory (~12 bytes/unique vector).
 *
 * @lucene.experimental
 */
final class DedupFlatVectorsWriter extends FlatVectorsWriter {

  private static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

  private final SegmentWriteState segmentWriteState;
  private final IndexOutput meta, data;
  private final IndexOutput tempMappings;

  /** Temp file for dictionary vectors — supports read-back for collision verification. */
  private IndexOutput dictTempOut;

  private final List<FieldWriter<?>> fields = new ArrayList<>();

  /** Deferred per-field metadata, written to meta/data in finish(). */
  private final List<PendingField> pendingFields = new ArrayList<>();

  /** Per-field dictionary: 64-bit hash → dictOrd. Reset per field. */
  private final LongIntHashMap hashToOrd = new LongIntHashMap();

  private int dictSize; // unique vectors in current field's dictionary
  private long dictDataOffset; // start of current field's dictionary in .dvd
  private int dictDimension; // dimension of current field
  private VectorEncoding dictEncoding; // encoding of current field
  private int vectorByteSize; // bytes per vector in current field
  private ByteBuffer writeBuf; // reused for writing float vectors
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
      tempMappings = state.directory.createTempOutput(dataFileName, "dedup-maps", state.context);

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

      dictDataOffset = data.getFilePointer();
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

  // --- Flush ---

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
              0,
              0,
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

    long mapOffset = tempMappings.getFilePointer();
    for (int newOrd = 0; newOrd < size; newOrd++) {
      int srcOrd = ordMap != null ? ordMap[newOrd] : newOrd;
      tempMappings.writeVInt(addToDict(vectors.get(srcOrd)));
    }
    long mapLength = tempMappings.getFilePointer() - mapOffset;

    pendingFields.add(
        new PendingField(
            field.fieldInfo,
            size,
            maxDoc,
            docsWithField,
            mapOffset,
            mapLength,
            dictSize,
            dictDataOffset,
            dictDimension,
            dictEncoding));
  }

  // --- Merge (streaming) ---

  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    initDict(fieldInfo);
    int maxDoc = segmentWriteState.segmentInfo.maxDoc();
    KnnVectorValues merged =
        switch (fieldInfo.getVectorEncoding()) {
          case FLOAT32 ->
              KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
          case BYTE ->
              KnnVectorsWriter.MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState);
        };
    streamMerge(fieldInfo, merged, maxDoc);
  }

  /**
   * Stream vectors one at a time: hash-dedup to dict, write mapping to temp. O(1) vector memory.
   */
  private void streamMerge(FieldInfo fieldInfo, KnnVectorValues values, int maxDoc)
      throws IOException {
    DocsWithFieldSet docsWithField = new DocsWithFieldSet();
    long mapOffset = tempMappings.getFilePointer();

    KnnVectorValues.DocIndexIterator iter = values.iterator();
    for (int doc = iter.nextDoc(); doc != NO_MORE_DOCS; doc = iter.nextDoc()) {
      Object vec =
          values instanceof FloatVectorValues fv
              ? fv.vectorValue(iter.index())
              : ((ByteVectorValues) values).vectorValue(iter.index());
      tempMappings.writeVInt(addToDict(vec));
      docsWithField.add(doc);
    }
    long mapLength = tempMappings.getFilePointer() - mapOffset;

    pendingFields.add(
        new PendingField(
            fieldInfo,
            docsWithField.cardinality(),
            maxDoc,
            docsWithField,
            mapOffset,
            mapLength,
            dictSize,
            dictDataOffset,
            dictDimension,
            dictEncoding));
  }

  @Override
  public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(
      FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    mergeOneField(fieldInfo, mergeState);
    // Re-stream to build scorer for HNSW (same pattern as Lucene99FlatVectorsWriter)
    KnnVectorValues merged =
        switch (fieldInfo.getVectorEncoding()) {
          case FLOAT32 ->
              KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
          case BYTE ->
              KnnVectorsWriter.MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState);
        };
    return buildTempFileScorer(fieldInfo, merged);
  }

  private CloseableRandomVectorScorerSupplier buildTempFileScorer(
      FieldInfo fieldInfo, KnnVectorValues values) throws IOException {
    int dim = fieldInfo.getVectorDimension();
    VectorEncoding encoding = fieldInfo.getVectorEncoding();
    int byteSize = dim * encoding.byteSize;

    // Write vectors to temp file
    IndexOutput tempOut =
        segmentWriteState.directory.createTempOutput(
            data.getName(), "dedup-scorer", segmentWriteState.context);
    int count = 0;
    IndexInput tempIn = null;
    try {
      ByteBuffer buf =
          encoding == VectorEncoding.FLOAT32
              ? ByteBuffer.allocate(byteSize).order(ByteOrder.LITTLE_ENDIAN)
              : null;
      KnnVectorValues.DocIndexIterator iter = values.iterator();
      for (int doc = iter.nextDoc(); doc != NO_MORE_DOCS; doc = iter.nextDoc()) {
        if (values instanceof FloatVectorValues fv) {
          buf.clear();
          buf.asFloatBuffer().put(fv.vectorValue(iter.index()));
          tempOut.writeBytes(buf.array(), byteSize);
        } else {
          byte[] v = ((ByteVectorValues) values).vectorValue(iter.index());
          tempOut.writeBytes(v, v.length);
        }
        count++;
      }
      CodecUtil.writeFooter(tempOut);
      IOUtils.close(tempOut);

      tempIn = segmentWriteState.directory.openInput(tempOut.getName(), segmentWriteState.context);

      KnnVectorValues vectorValues =
          switch (encoding) {
            case FLOAT32 ->
                new OffHeapFloatVectorValues.DenseOffHeapVectorValues(
                    dim,
                    count,
                    tempIn,
                    byteSize,
                    vectorsScorer,
                    fieldInfo.getVectorSimilarityFunction());
            case BYTE ->
                new OffHeapByteVectorValues.DenseOffHeapVectorValues(
                    dim,
                    count,
                    tempIn,
                    byteSize,
                    vectorsScorer,
                    fieldInfo.getVectorSimilarityFunction());
          };

      RandomVectorScorerSupplier supplier =
          vectorsScorer.getRandomVectorScorerSupplier(
              fieldInfo.getVectorSimilarityFunction(), vectorValues);

      final int finalCount = count;
      final IndexInput finalTempIn = tempIn;
      final String tempFileName = tempOut.getName();
      tempIn = null; // ownership transferred to the supplier
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
      IOUtils.closeWhileSuppressingExceptions(t, tempIn, tempOut);
      IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, tempOut.getName());
      throw t;
    }
  }

  // --- Dictionary ---

  private void initDict(FieldInfo fieldInfo) throws IOException {
    int dim = fieldInfo.getVectorDimension();
    VectorEncoding enc = fieldInfo.getVectorEncoding();

    // Reuse existing dictionary for cross-field dedup when dim/encoding match
    if (dictTempName != null && dim == dictDimension && enc == dictEncoding) {
      // Ensure the dict temp is writable (may have been closed for collision read-back)
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
    if (enc == VectorEncoding.FLOAT32) {
      writeBuf = ByteBuffer.allocate(vectorByteSize).order(ByteOrder.LITTLE_ENDIAN);
    }
    dictTempOut =
        segmentWriteState.directory.createTempOutput(
            data.getName(), "dedup-dict", segmentWriteState.context);
    dictTempName = dictTempOut.getName();
  }

  /** Copy dictionary vectors from temp file to the main data file. */
  private void copyDictTempToData() throws IOException {
    if (dictTempName == null) return;
    // Close whichever handle is open
    if (dictReadHandle != null) {
      // Reader was open (from collision check) — writer already has footer
      IOUtils.close(dictReadHandle);
      dictReadHandle = null;
    } else if (dictTempOut != null) {
      // Writer is still open — close it with footer
      CodecUtil.writeFooter(dictTempOut);
      IOUtils.close(dictTempOut);
    }
    // dictTempOut may have been closed by getDictReadHandle without nulling
    dictTempOut = null;
    try (IndexInput in =
        segmentWriteState.directory.openInput(dictTempName, segmentWriteState.context)) {
      data.copyBytes(in, in.length() - CodecUtil.footerLength());
    }
    IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, dictTempName);
    dictTempName = null;
  }

  /**
   * Add vector to dictionary. On hash hit, reads back from the dictionary file to verify equality.
   * On collision (same hash, different vector), linear-probes the hash space. Returns dictOrd.
   */
  private <T> int addToDict(T vec) throws IOException {
    long hash = vectorHash(vec);
    // Linear probe: on collision, try hash+1, hash+2, etc.
    for (int probe = 0; ; probe++) {
      long probeHash = hash + probe;
      int existing = hashToOrd.getOrDefault(probeHash, -1);
      if (existing < 0) {
        // New entry — write vector to dictionary
        ensureDictWritable();
        int dictOrd = dictSize++;
        hashToOrd.put(probeHash, dictOrd);
        writeVectorToDict(vec);
        return dictOrd;
      }
      // Hash hit — verify equality by reading back from dictionary file
      if (vectorMatchesDict(vec, existing)) {
        return existing;
      }
      // Hash collision with different vector — continue probing
    }
  }

  /** Read vector at dictOrd from the dictionary temp file and compare to vec. */
  private <T> boolean vectorMatchesDict(T vec, int dictOrd) throws IOException {
    // dictOrd is relative to the current field's dictionary in the temp file
    long offset = (long) dictOrd * vectorByteSize;
    IndexInput readHandle = getDictReadHandle();
    if (vec instanceof float[] f) {
      byte[] stored = new byte[vectorByteSize];
      readHandle.seek(offset);
      readHandle.readBytes(stored, 0, vectorByteSize);
      ByteBuffer storedBuf = ByteBuffer.wrap(stored).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < f.length; i++) {
        if (Float.floatToRawIntBits(f[i]) != Float.floatToRawIntBits(storedBuf.getFloat())) {
          return false;
        }
      }
      return true;
    } else if (vec instanceof byte[] b) {
      byte[] stored = new byte[vectorByteSize];
      readHandle.seek(offset);
      readHandle.readBytes(stored, 0, vectorByteSize);
      return Arrays.equals(b, 0, b.length, stored, 0, stored.length);
    }
    return false;
  }

  /** Lazily opened read handle for the dictionary temp file (for collision verification). */
  private IndexInput dictReadHandle;

  private String dictTempName; // name of the current dict temp file

  private IndexInput getDictReadHandle() throws IOException {
    if (dictReadHandle == null) {
      // Close the write handle so we can open for reading (MockDirectoryWrapper requirement)
      // We'll reopen for writing after the comparison
      CodecUtil.writeFooter(dictTempOut);
      IOUtils.close(dictTempOut);
      dictTempOut = null; // mark as closed
      dictReadHandle =
          segmentWriteState.directory.openInput(dictTempName, segmentWriteState.context);
    }
    return dictReadHandle;
  }

  /** After collision verification, reopen the dict temp for writing if it was closed. */
  private void ensureDictWritable() throws IOException {
    if (dictReadHandle != null) {
      IOUtils.close(dictReadHandle);
      dictReadHandle = null;
      // Reopen: copy existing content to a new temp, then continue writing
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

  private <T> void writeVectorToDict(T vec) throws IOException {
    if (vec instanceof float[] f) {
      writeBuf.clear();
      writeBuf.asFloatBuffer().put(f);
      dictTempOut.writeBytes(writeBuf.array(), vectorByteSize);
    } else if (vec instanceof byte[] b) {
      dictTempOut.writeBytes(b, b.length);
    }
  }

  /**
   * 64-bit hash combining {@link Arrays#hashCode} (upper 32 bits) with FNV-1a (lower 32 bits).
   * Collisions are still handled correctly by {@link #vectorMatchesDict}, but a stronger hash
   * reduces the frequency of expensive read-back verification during merge.
   */
  static long vectorHash(Object vec) {
    if (vec instanceof float[] f) {
      return ((long) Arrays.hashCode(f) << 32) | (fnvHashFloat(f) & 0xFFFFFFFFL);
    } else if (vec instanceof byte[] b) {
      return ((long) Arrays.hashCode(b) << 32) | (fnvHashByte(b) & 0xFFFFFFFFL);
    }
    throw new IllegalArgumentException("Unsupported vector type");
  }

  private static int fnvHashFloat(float[] f) {
    int h = 0x811c9dc5;
    for (float v : f) {
      int bits = Float.floatToRawIntBits(v);
      h = (h ^ (bits & 0xFF)) * 0x01000193;
      h = (h ^ ((bits >>> 8) & 0xFF)) * 0x01000193;
      h = (h ^ ((bits >>> 16) & 0xFF)) * 0x01000193;
      h = (h ^ ((bits >>> 24) & 0xFF)) * 0x01000193;
    }
    return h;
  }

  private static int fnvHashByte(byte[] b) {
    int h = 0x811c9dc5;
    for (byte v : b) {
      h = (h ^ (v & 0xFF)) * 0x01000193;
    }
    return h;
  }

  // --- finish / close ---

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

    // Dictionary is complete in `data`. Now append mappings from temp file.
    CodecUtil.writeFooter(tempMappings);
    IOUtils.close(tempMappings);

    long mappingsStartInData = data.getFilePointer();
    try (IndexInput mappingsIn =
        segmentWriteState.directory.openInput(tempMappings.getName(), segmentWriteState.context)) {
      data.copyBytes(mappingsIn, mappingsIn.length() - CodecUtil.footerLength());
    }

    // Write per-field metadata (including per-field dict info) to meta, OrdToDoc DISI to data
    for (PendingField pf : pendingFields) {
      meta.writeInt(pf.fieldInfo.number);
      meta.writeVInt(pf.size);
      meta.writeInt(pf.fieldInfo.getVectorSimilarityFunction().ordinal());
      meta.writeVInt(pf.dictSize);
      meta.writeVInt(pf.dictDimension);
      meta.writeVInt(pf.dictEncoding != null ? pf.dictEncoding.ordinal() : -1);
      meta.writeVLong(pf.dictDataOffset);
      meta.writeVLong(mappingsStartInData + pf.mapOffsetInTemp);
      meta.writeVLong(pf.mapLength);
      OrdToDocDISIReaderConfiguration.writeStoredMeta(
          DIRECT_MONOTONIC_BLOCK_SHIFT, meta, data, pf.size, pf.maxDoc, pf.docsWithField);
    }

    meta.writeInt(-1); // end-of-fields sentinel

    CodecUtil.writeFooter(meta);
    CodecUtil.writeFooter(data);

    IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, tempMappings.getName());
  }

  @Override
  public long ramBytesUsed() {
    long total = RamUsageEstimator.shallowSizeOfInstance(DedupFlatVectorsWriter.class);
    for (FieldWriter<?> field : fields) total += field.ramBytesUsed();
    for (PendingField pf : pendingFields) total += pf.docsWithField.ramBytesUsed();
    total += hashToOrd.size() * 16L;
    return total;
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, data, tempMappings, dictReadHandle, dictTempOut);
    IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, tempMappings.getName());
    if (dictTempName != null) {
      IOUtils.deleteFilesIgnoringExceptions(segmentWriteState.directory, dictTempName);
    }
  }

  // --- Records ---

  private record PendingField(
      FieldInfo fieldInfo,
      int size,
      int maxDoc,
      DocsWithFieldSet docsWithField,
      long mapOffsetInTemp,
      long mapLength,
      int dictSize,
      long dictDataOffset,
      int dictDimension,
      VectorEncoding dictEncoding) {}

  /** Buffers vectors in memory during indexing. */
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
      if (docID == lastDocID)
        throw new IllegalArgumentException(
            "VectorValuesField \"" + fieldInfo.name + "\" appears more than once in this document");
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
