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
import java.util.Map;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.codecs.lucene95.OffHeapByteVectorValues;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;

/**
 * Reader for the dedup vector format. Each field's vectors are backed by a shared dictionary region
 * in the .dvd file. Fields with no duplicates delegate to {@link OffHeapFloatVectorValues} for
 * zero-overhead search. Fields with duplicates use {@link DedupFloatVectorValues} which implements
 * {@link HasIndexSlice} and overrides {@link HasIndexSlice#ordToOffset(int, int)} to enable
 * off-heap SIMD scoring through the ordinal indirection.
 *
 * @lucene.experimental
 */
final class DedupFlatVectorsReader extends FlatVectorsReader {

  private final IntObjectHashMap<FieldEntry> fields = new IntObjectHashMap<>();
  private final FieldInfos fieldInfos;
  private final IndexInput data;

  /**
   * IO context used to open {@link #data}. Captured so that {@link #getMergeInstance()} can switch
   * the underlying access hint to {@link DataAccessHint#SEQUENTIAL} for the duration of a merge,
   * and {@link #finishMerge()} can revert it back to the original context for search.
   */
  private final IOContext dataContext;

  DedupFlatVectorsReader(SegmentReadState state, FlatVectorsScorer scorer) throws IOException {
    super(scorer);
    this.fieldInfos = state.fieldInfos;

    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, DedupFlatVectorsFormat.META_EXTENSION);
    int versionMeta = readMeta(state, metaFileName);

    String dataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, DedupFlatVectorsFormat.DATA_EXTENSION);
    this.dataContext = state.context.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS);
    IndexInput dataIn = state.directory.openInput(dataFileName, dataContext);
    try {
      int versionData =
          CodecUtil.checkIndexHeader(
              dataIn,
              DedupFlatVectorsFormat.DATA_CODEC_NAME,
              DedupFlatVectorsFormat.VERSION_START,
              DedupFlatVectorsFormat.VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      if (versionMeta != versionData) {
        throw new CorruptIndexException(
            "Format versions mismatch: meta=" + versionMeta + ", data=" + versionData, dataIn);
      }
      CodecUtil.retrieveChecksum(dataIn);
      this.data = dataIn;
      loadAllOrdToDict();
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, dataIn);
      throw t;
    }
  }

  /** Read .dvm metadata. Returns the codec version. */
  private int readMeta(SegmentReadState state, String metaFileName) throws IOException {
    int versionMeta = -1;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
      Throwable priorE = null;
      try {
        versionMeta =
            CodecUtil.checkIndexHeader(
                meta,
                DedupFlatVectorsFormat.META_CODEC_NAME,
                DedupFlatVectorsFormat.VERSION_START,
                DedupFlatVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix);
        readFieldEntries(meta);
      } catch (Throwable exception) {
        priorE = exception;
      } finally {
        CodecUtil.checkFooter(meta, priorE);
      }
    }
    return versionMeta;
  }

  /** Parse per-field entries from the .dvm meta stream. */
  private void readFieldEntries(ChecksumIndexInput meta) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      int size = meta.readVInt();
      VectorSimilarityFunction simFunc = VectorSimilarityFunction.values()[meta.readInt()];
      int dictSize = meta.readVInt();
      int dictDimension = meta.readVInt();
      int encOrd = meta.readVInt();
      VectorEncoding encoding = encOrd >= 0 ? VectorEncoding.values()[encOrd] : null;
      long dictDataOffset = meta.readVLong();
      long mapOffset = meta.readVLong();
      long mapLength = meta.readVLong();
      OrdToDocDISIReaderConfiguration ordToDoc =
          OrdToDocDISIReaderConfiguration.fromStoredMeta(meta, size);

      int vectorByteSize = encoding != null ? dictDimension * encoding.byteSize : 0;
      fields.put(
          info.number,
          new FieldEntry(
              size,
              simFunc,
              dictSize,
              dictDimension,
              encoding,
              dictDataOffset,
              vectorByteSize,
              mapOffset,
              mapLength,
              ordToDoc,
              null));
    }
  }

  /**
   * Eagerly load all ordToDict mappings from the .dvd file into the corresponding FieldEntry.
   * Called once at construction time after the data file is opened.
   */
  private void loadAllOrdToDict() throws IOException {
    for (var cursor : fields) {
      FieldEntry entry = cursor.value;
      if (entry.size > 0 && entry.mapLength > 0) {
        int[] ordToDict = new int[entry.size];
        IndexInput slice = data.slice("dedup-map", entry.mapOffset, entry.mapLength);
        for (int i = 0; i < entry.size; i++) {
          ordToDict[i] = slice.readVInt();
        }
        // Replace the entry with one that includes the loaded mapping
        fields.put(cursor.key, entry.withOrdToDict(ordToDict));
      }
    }
  }

  // ---- Public API ----

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    FieldEntry entry = getFieldEntry(field);
    if (entry.ordToDict == null) {
      // No duplicates — delegate to the standard off-heap implementation (zero overhead)
      return OffHeapFloatVectorValues.load(
          entry.simFunc,
          vectorScorer,
          entry.ordToDoc,
          entry.encoding,
          entry.dictDimension,
          entry.dictDataOffset,
          entry.dictDataLength(),
          data);
    }
    return new DedupFloatVectorValues(entry, data, vectorScorer);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    FieldEntry entry = getFieldEntry(field);
    if (entry.ordToDict == null) {
      return OffHeapByteVectorValues.load(
          entry.simFunc,
          vectorScorer,
          entry.ordToDoc,
          entry.encoding,
          entry.dictDimension,
          entry.dictDataOffset,
          entry.dictDataLength(),
          data);
    }
    return new DedupByteVectorValues(entry, data, vectorScorer);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
    FieldEntry entry = getFieldEntry(field);
    return vectorScorer.getRandomVectorScorer(entry.simFunc, getFloatVectorValues(field), target);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
    FieldEntry entry = getFieldEntry(field);
    return vectorScorer.getRandomVectorScorer(entry.simFunc, getByteVectorValues(field), target);
  }

  // ---- Package-private accessors for merge-path optimization ----
  //
  // These let a dedup-aware writer reuse this reader's existing dictionary + ordToDict mapping
  // instead of hashing each per-doc vector individually at merge time. See
  // DedupFlatVectorsWriter#buildFloatSubs / #buildByteSubs.

  /**
   * Return the per-doc {@code ordToDict} mapping for the given field, or {@code null} if the field
   * has no duplicates (identity mapping — no value in trying to skip hashing since there are no
   * duplicates to collapse).
   */
  int[] getOrdToDict(String field) {
    return getFieldEntry(field).ordToDict;
  }

  /**
   * Return a random-access view over the {@code dictSize} unique float vectors in the field's
   * dictionary, indexed by source dict ord. The merge writer walks this once, hashes each vector
   * into its own target dict, and builds a {@code sourceDictOrd → targetDictOrd} mapping — which it
   * then uses to resolve every per-doc ord with a single array lookup (no per-doc hashing or
   * equality check).
   *
   * <p>Returns {@code null} if the field's encoding is not float32.
   */
  FloatVectorValues getDictFloatVectorValues(String field) throws IOException {
    FieldEntry entry = getFieldEntry(field);
    if (entry.encoding != VectorEncoding.FLOAT32) {
      return null;
    }
    IndexInput dictSlice = data.slice("dedup-dict", entry.dictDataOffset, entry.dictDataLength());
    return new OffHeapFloatVectorValues.DenseOffHeapVectorValues(
        entry.dictDimension,
        entry.dictSize,
        dictSlice,
        entry.vectorByteSize,
        vectorScorer,
        entry.simFunc);
  }

  /** Byte-encoded mirror of {@link #getDictFloatVectorValues(String)}. */
  ByteVectorValues getDictByteVectorValues(String field) throws IOException {
    FieldEntry entry = getFieldEntry(field);
    if (entry.encoding != VectorEncoding.BYTE) {
      return null;
    }
    IndexInput dictSlice = data.slice("dedup-dict", entry.dictDataOffset, entry.dictDataLength());
    return new OffHeapByteVectorValues.DenseOffHeapVectorValues(
        entry.dictDimension,
        entry.dictSize,
        dictSlice,
        entry.vectorByteSize,
        vectorScorer,
        entry.simFunc);
  }

  @Override
  public void checkIntegrity() throws IOException {
    CodecUtil.checksumEntireFile(data);
  }

  @Override
  public FlatVectorsReader getMergeInstance() throws IOException {
    // During merge, the dedup writer walks each source segment's vectors mostly in forward order
    // (via DocIDMerger). Switch the IO hint to SEQUENTIAL so the directory can enable readahead.
    // Occasional random-access reads for hash-collision verification target recently-read vectors,
    // which typically remain resident in the page cache, so they don't materially suffer from the
    // SEQUENTIAL hint.
    data.updateIOContext(dataContext.withHints(DataAccessHint.SEQUENTIAL));
    return this;
  }

  @Override
  public void finishMerge() throws IOException {
    // Revert the access hint to the original context used for search.
    data.updateIOContext(dataContext);
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(data);
  }

  @Override
  public long ramBytesUsed() {
    long total = fields.ramBytesUsed();
    for (var cursor : fields) {
      FieldEntry entry = cursor.value;
      if (entry.ordToDict != null) {
        total += (long) entry.ordToDict.length * Integer.BYTES;
      }
    }
    return total;
  }

  @Override
  public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
    FieldEntry entry = fields.get(fieldInfo.number);
    if (entry == null) return Map.of();
    return Map.of("vec", (long) entry.size * entry.vectorByteSize);
  }

  private FieldEntry getFieldEntry(String field) {
    FieldInfo info = fieldInfos.fieldInfo(field);
    if (info == null) {
      throw new IllegalArgumentException("field=\"" + field + "\" not found");
    }
    FieldEntry entry = fields.get(info.number);
    if (entry == null) {
      throw new IllegalArgumentException("field=\"" + field + "\" has no dedup entry");
    }
    return entry;
  }

  // ---- Inner types ----

  /**
   * Per-field metadata read from .dvm. The {@code ordToDict} array is null until eagerly loaded
   * from .dvd at construction time, and remains null for fields with no duplicates (identity
   * mapping).
   */
  private record FieldEntry(
      int size,
      VectorSimilarityFunction simFunc,
      int dictSize,
      int dictDimension,
      VectorEncoding encoding,
      long dictDataOffset,
      int vectorByteSize,
      long mapOffset,
      long mapLength,
      OrdToDocDISIReaderConfiguration ordToDoc,
      int[] ordToDict) {

    /** Total byte length of the dictionary region for this field. */
    long dictDataLength() {
      return (long) dictSize * vectorByteSize;
    }

    /** Return a copy of this entry with the ordToDict mapping populated. */
    FieldEntry withOrdToDict(int[] ordToDict) {
      return new FieldEntry(
          size,
          simFunc,
          dictSize,
          dictDimension,
          encoding,
          dictDataOffset,
          vectorByteSize,
          mapOffset,
          mapLength,
          ordToDoc,
          ordToDict);
    }
  }

  /**
   * Float vector values backed by a shared dictionary with ordinal indirection. Implements {@link
   * HasIndexSlice} so the MemorySegment scorer can operate directly on the memory-mapped dictionary
   * using {@link #ordToOffset(int, int)} for address translation.
   */
  private static final class DedupFloatVectorValues extends FloatVectorValues
      implements HasIndexSlice {

    private final int size;
    private final int dimension;
    private final int vectorByteSize;
    private final int[] ordToDict;
    private final VectorSimilarityFunction simFunc;
    private final OrdToDocDISIReaderConfiguration ordToDocConfig;
    private final FlatVectorsScorer flatScorer;
    private final IndexInput dictSlice;
    private final DirectMonotonicReader ordToDocReader;
    private final float[] value;
    private int lastOrd = -1;

    DedupFloatVectorValues(FieldEntry entry, IndexInput data, FlatVectorsScorer flatScorer)
        throws IOException {
      this(
          entry.size,
          entry.dictDimension,
          entry.vectorByteSize,
          entry.ordToDict,
          entry.simFunc,
          entry.ordToDoc,
          flatScorer,
          data.slice("dedup-dict", entry.dictDataOffset, entry.dictDataLength()),
          (!entry.ordToDoc.isDense() && !entry.ordToDoc.isEmpty())
              ? entry.ordToDoc.getDirectMonotonicReader(data)
              : null);
    }

    private DedupFloatVectorValues(
        int size,
        int dimension,
        int vectorByteSize,
        int[] ordToDict,
        VectorSimilarityFunction simFunc,
        OrdToDocDISIReaderConfiguration ordToDocConfig,
        FlatVectorsScorer flatScorer,
        IndexInput dictSlice,
        DirectMonotonicReader ordToDocReader) {
      this.size = size;
      this.dimension = dimension;
      this.vectorByteSize = vectorByteSize;
      this.ordToDict = ordToDict;
      this.simFunc = simFunc;
      this.ordToDocConfig = ordToDocConfig;
      this.flatScorer = flatScorer;
      this.dictSlice = dictSlice;
      this.ordToDocReader = ordToDocReader;
      this.value = new float[dimension];
    }

    @Override
    public IndexInput getSlice() {
      return dictSlice;
    }

    @Override
    public long ordToOffset(int ord, int vectorByteSize) {
      return (long) ordToDict[ord] * vectorByteSize;
    }

    @Override
    public int dimension() {
      return dimension;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
      if (ord == lastOrd) return value;
      dictSlice.seek(ordToOffset(ord, vectorByteSize));
      dictSlice.readFloats(value, 0, dimension);
      lastOrd = ord;
      return value;
    }

    @Override
    public DocIndexIterator iterator() {
      return ordToDocConfig.isDense() ? createDenseIterator() : createSparseIterator();
    }

    @Override
    public int ordToDoc(int ord) {
      return ordToDocReader != null ? (int) ordToDocReader.get(ord) : ord;
    }

    @Override
    public FloatVectorValues copy() throws IOException {
      return new DedupFloatVectorValues(
          size,
          dimension,
          vectorByteSize,
          ordToDict,
          simFunc,
          ordToDocConfig,
          flatScorer,
          dictSlice.clone(),
          ordToDocReader);
    }

    @Override
    public VectorScorer scorer(float[] query) throws IOException {
      if (size == 0) return null;
      DedupFloatVectorValues copy = (DedupFloatVectorValues) copy();
      DocIndexIterator iter = copy.iterator();
      RandomVectorScorer rs = flatScorer.getRandomVectorScorer(simFunc, copy, query);
      return new VectorScorer() {
        @Override
        public float score() throws IOException {
          return rs.score(iter.index());
        }

        @Override
        public DocIdSetIterator iterator() {
          return iter;
        }
      };
    }
  }

  /**
   * Byte vector values backed by a shared dictionary with ordinal indirection. Mirror of {@link
   * DedupFloatVectorValues} for byte-encoded vectors.
   */
  private static final class DedupByteVectorValues extends ByteVectorValues
      implements HasIndexSlice {

    private final int size;
    private final int dimension;
    private final int vectorByteSize;
    private final int[] ordToDict;
    private final VectorSimilarityFunction simFunc;
    private final OrdToDocDISIReaderConfiguration ordToDocConfig;
    private final FlatVectorsScorer flatScorer;
    private final IndexInput dictSlice;
    private final DirectMonotonicReader ordToDocReader;
    private final byte[] value;
    private int lastOrd = -1;

    DedupByteVectorValues(FieldEntry entry, IndexInput data, FlatVectorsScorer flatScorer)
        throws IOException {
      this(
          entry.size,
          entry.dictDimension,
          entry.vectorByteSize,
          entry.ordToDict,
          entry.simFunc,
          entry.ordToDoc,
          flatScorer,
          data.slice("dedup-dict", entry.dictDataOffset, entry.dictDataLength()),
          (!entry.ordToDoc.isDense() && !entry.ordToDoc.isEmpty())
              ? entry.ordToDoc.getDirectMonotonicReader(data)
              : null);
    }

    private DedupByteVectorValues(
        int size,
        int dimension,
        int vectorByteSize,
        int[] ordToDict,
        VectorSimilarityFunction simFunc,
        OrdToDocDISIReaderConfiguration ordToDocConfig,
        FlatVectorsScorer flatScorer,
        IndexInput dictSlice,
        DirectMonotonicReader ordToDocReader) {
      this.size = size;
      this.dimension = dimension;
      this.vectorByteSize = vectorByteSize;
      this.ordToDict = ordToDict;
      this.simFunc = simFunc;
      this.ordToDocConfig = ordToDocConfig;
      this.flatScorer = flatScorer;
      this.dictSlice = dictSlice;
      this.ordToDocReader = ordToDocReader;
      this.value = new byte[dimension];
    }

    @Override
    public IndexInput getSlice() {
      return dictSlice;
    }

    @Override
    public long ordToOffset(int ord, int vectorByteSize) {
      return (long) ordToDict[ord] * vectorByteSize;
    }

    @Override
    public int dimension() {
      return dimension;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public byte[] vectorValue(int ord) throws IOException {
      if (ord == lastOrd) return value;
      dictSlice.seek(ordToOffset(ord, vectorByteSize));
      dictSlice.readBytes(value, 0, dimension);
      lastOrd = ord;
      return value;
    }

    @Override
    public DocIndexIterator iterator() {
      return ordToDocConfig.isDense() ? createDenseIterator() : createSparseIterator();
    }

    @Override
    public int ordToDoc(int ord) {
      return ordToDocReader != null ? (int) ordToDocReader.get(ord) : ord;
    }

    @Override
    public ByteVectorValues copy() throws IOException {
      return new DedupByteVectorValues(
          size,
          dimension,
          vectorByteSize,
          ordToDict,
          simFunc,
          ordToDocConfig,
          flatScorer,
          dictSlice.clone(),
          ordToDocReader);
    }

    @Override
    public VectorScorer scorer(byte[] query) throws IOException {
      if (size == 0) return null;
      DedupByteVectorValues copy = (DedupByteVectorValues) copy();
      DocIndexIterator iter = copy.iterator();
      RandomVectorScorer rs = flatScorer.getRandomVectorScorer(simFunc, copy, query);
      return new VectorScorer() {
        @Override
        public float score() throws IOException {
          return rs.score(iter.index());
        }

        @Override
        public DocIdSetIterator iterator() {
          return iter;
        }
      };
    }
  }
}
