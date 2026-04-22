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
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;

/** Reader for the dedup vector format with per-field dictionary metadata. */
final class DedupFlatVectorsReader extends FlatVectorsReader {

  private final IntObjectHashMap<FieldEntry> fields = new IntObjectHashMap<>();

  /** Eagerly loaded ordToDict mappings, keyed by field number. */
  private final IntObjectHashMap<int[]> ordToDictCache = new IntObjectHashMap<>();

  private final FieldInfos fieldInfos;
  private final IndexInput data;

  DedupFlatVectorsReader(SegmentReadState state, FlatVectorsScorer scorer) throws IOException {
    super(scorer);
    this.fieldInfos = state.fieldInfos;

    int versionMeta = -1;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, DedupFlatVectorsFormat.META_EXTENSION);
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
        readFields(meta, state.fieldInfos);
      } catch (Throwable exception) {
        priorE = exception;
      } finally {
        CodecUtil.checkFooter(meta, priorE);
      }
    }

    String dataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, DedupFlatVectorsFormat.DATA_EXTENSION);
    IndexInput dataIn = state.directory.openInput(dataFileName, state.context);
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

      // Eagerly load ordToDict mappings now that data file is available
      loadAllOrdToDict();
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, dataIn);
      throw t;
    }
  }

  private void readFields(ChecksumIndexInput meta, FieldInfos infos) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = infos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      int size = meta.readVInt();
      VectorSimilarityFunction simFunc = VectorSimilarityFunction.values()[meta.readInt()];
      int dictSize = meta.readVInt();
      int dictDimension = meta.readVInt();
      int encOrd = meta.readVInt();
      VectorEncoding dictEncoding = encOrd >= 0 ? VectorEncoding.values()[encOrd] : null;
      long dictDataOffset = meta.readVLong();
      long mapOffset = meta.readVLong();
      long mapLength = meta.readVLong();
      OrdToDocDISIReaderConfiguration ordToDoc =
          OrdToDocDISIReaderConfiguration.fromStoredMeta(meta, size);

      int vectorByteSize = dictEncoding != null ? dictDimension * dictEncoding.byteSize : 0;
      fields.put(
          info.number,
          new FieldEntry(
              size,
              simFunc,
              dictSize,
              dictDimension,
              dictEncoding,
              dictDataOffset,
              vectorByteSize,
              mapOffset,
              mapLength,
              ordToDoc));
    }
  }

  /** Load all ordToDict mappings eagerly. Called once after data file is opened. */
  private void loadAllOrdToDict() throws IOException {
    for (var cursor : fields) {
      FieldEntry entry = cursor.value;
      if (entry.size > 0 && entry.mapLength > 0) {
        int[] ordToDict = new int[entry.size];
        IndexInput slice = data.slice("dedup-map", entry.mapOffset, entry.mapLength);
        for (int i = 0; i < entry.size; i++) {
          ordToDict[i] = slice.readVInt();
        }
        ordToDictCache.put(cursor.key, ordToDict);
      }
    }
  }

  /** Get the cached ordToDict for a field, or null if identity mapping. */
  private int[] getOrdToDict(int fieldNumber) {
    return ordToDictCache.get(fieldNumber);
  }

  private FieldEntry getFieldEntry(String field) {
    FieldInfo info = fieldInfos.fieldInfo(field);
    if (info == null) throw new IllegalArgumentException("field=\"" + field + "\" not found");
    FieldEntry entry = fields.get(info.number);
    if (entry == null) throw new IllegalArgumentException("field=\"" + field + "\" has no entry");
    return entry;
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    FieldInfo info = fieldInfos.fieldInfo(field);
    FieldEntry entry = getFieldEntry(field);
    int[] ordToDict = getOrdToDict(info.number);
    if (ordToDict == null) {
      // Identity mapping — use the same off-heap implementation as the base format
      long vectorDataLength = (long) entry.dictSize * entry.vectorByteSize;
      return OffHeapFloatVectorValues.load(
          entry.simFunc,
          vectorScorer,
          entry.ordToDoc,
          entry.dictEncoding,
          entry.dictDimension,
          entry.dictDataOffset,
          vectorDataLength,
          data);
    }
    return new DedupFloatVectorValues(entry, ordToDict, data, vectorScorer);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    FieldInfo info = fieldInfos.fieldInfo(field);
    FieldEntry entry = getFieldEntry(field);
    int[] ordToDict = getOrdToDict(info.number);
    if (ordToDict == null) {
      long vectorDataLength = (long) entry.dictSize * entry.vectorByteSize;
      return OffHeapByteVectorValues.load(
          entry.simFunc,
          vectorScorer,
          entry.ordToDoc,
          entry.dictEncoding,
          entry.dictDimension,
          entry.dictDataOffset,
          vectorDataLength,
          data);
    }
    return new DedupByteVectorValues(entry, ordToDict, data, vectorScorer);
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

  @Override
  public void checkIntegrity() throws IOException {
    CodecUtil.checksumEntireFile(data);
  }

  @Override
  public FlatVectorsReader getMergeInstance() {
    return this;
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(data);
  }

  @Override
  public long ramBytesUsed() {
    long total = fields.ramBytesUsed();
    for (var cursor : ordToDictCache) {
      if (cursor.value != null) {
        total += (long) cursor.value.length * Integer.BYTES;
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

  // --- Inner types ---

  private record FieldEntry(
      int size,
      VectorSimilarityFunction simFunc,
      int dictSize,
      int dictDimension,
      VectorEncoding dictEncoding,
      long dictDataOffset,
      int vectorByteSize,
      long mapOffset,
      long mapLength,
      OrdToDocDISIReaderConfiguration ordToDoc) {}

  private static final class DedupFloatVectorValues extends FloatVectorValues
      implements HasIndexSlice {
    private final int size;
    private final int dimension;
    private final long dictDataOffset;
    private final int vectorByteSize;
    private final int[] ordToDict;
    private final VectorSimilarityFunction simFunc;
    private final OrdToDocDISIReaderConfiguration ordToDocConfig;
    private final IndexInput dataSlice;
    private final float[] value;
    private int lastOrd = -1;
    private final DirectMonotonicReader ordToDocReader;
    private final FlatVectorsScorer flatScorer;

    DedupFloatVectorValues(
        FieldEntry entry, int[] ordToDict, IndexInput data, FlatVectorsScorer flatScorer)
        throws IOException {
      this.size = entry.size;
      this.dimension = entry.dictDimension;
      this.dictDataOffset = entry.dictDataOffset;
      this.vectorByteSize = entry.vectorByteSize;
      this.ordToDict = ordToDict;
      this.simFunc = entry.simFunc;
      this.ordToDocConfig = entry.ordToDoc;
      this.dataSlice =
          data.slice(
              "dedup-dict", entry.dictDataOffset, (long) entry.dictSize * entry.vectorByteSize);
      this.value = new float[dimension];
      this.flatScorer = flatScorer;
      this.ordToDocReader =
          (!ordToDocConfig.isDense() && !ordToDocConfig.isEmpty())
              ? ordToDocConfig.getDirectMonotonicReader(data)
              : null;
    }

    @Override
    public IndexInput getSlice() {
      return dataSlice;
    }

    @Override
    public long ordToOffset(int ord) {
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
      dataSlice.seek(ordToOffset(ord));
      dataSlice.readFloats(value, 0, dimension);
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
          dictDataOffset,
          vectorByteSize,
          ordToDict,
          simFunc,
          ordToDocConfig,
          dataSlice.clone(),
          flatScorer);
    }

    /** Copy constructor with direct fields. */
    private DedupFloatVectorValues(
        int size,
        int dimension,
        long dictDataOffset,
        int vectorByteSize,
        int[] ordToDict,
        VectorSimilarityFunction simFunc,
        OrdToDocDISIReaderConfiguration ordToDocConfig,
        IndexInput dataSlice,
        FlatVectorsScorer flatScorer)
        throws IOException {
      this.size = size;
      this.dimension = dimension;
      this.dictDataOffset = dictDataOffset;
      this.vectorByteSize = vectorByteSize;
      this.ordToDict = ordToDict;
      this.simFunc = simFunc;
      this.ordToDocConfig = ordToDocConfig;
      this.dataSlice = dataSlice;
      this.value = new float[dimension];
      this.flatScorer = flatScorer;
      this.ordToDocReader =
          (!ordToDocConfig.isDense() && !ordToDocConfig.isEmpty())
              ? ordToDocConfig.getDirectMonotonicReader(dataSlice)
              : null;
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

  private static final class DedupByteVectorValues extends ByteVectorValues
      implements HasIndexSlice {
    private final int size;
    private final int dimension;
    private final long dictDataOffset;
    private final int vectorByteSize;
    private final int[] ordToDict;
    private final VectorSimilarityFunction simFunc;
    private final OrdToDocDISIReaderConfiguration ordToDocConfig;
    private final IndexInput dataSlice;
    private final byte[] value;
    private int lastOrd = -1;
    private final DirectMonotonicReader ordToDocReader;
    private final FlatVectorsScorer flatScorer;

    DedupByteVectorValues(
        FieldEntry entry, int[] ordToDict, IndexInput data, FlatVectorsScorer flatScorer)
        throws IOException {
      this.size = entry.size;
      this.dimension = entry.dictDimension;
      this.dictDataOffset = entry.dictDataOffset;
      this.vectorByteSize = entry.vectorByteSize;
      this.ordToDict = ordToDict;
      this.simFunc = entry.simFunc;
      this.ordToDocConfig = entry.ordToDoc;
      this.dataSlice =
          data.slice(
              "dedup-dict", entry.dictDataOffset, (long) entry.dictSize * entry.vectorByteSize);
      this.value = new byte[dimension];
      this.flatScorer = flatScorer;
      this.ordToDocReader =
          (!ordToDocConfig.isDense() && !ordToDocConfig.isEmpty())
              ? ordToDocConfig.getDirectMonotonicReader(data)
              : null;
    }

    @Override
    public IndexInput getSlice() {
      return dataSlice;
    }

    @Override
    public long ordToOffset(int ord) {
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
      dataSlice.seek(ordToOffset(ord));
      dataSlice.readBytes(value, 0, dimension);
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
          dictDataOffset,
          vectorByteSize,
          ordToDict,
          simFunc,
          ordToDocConfig,
          dataSlice.clone(),
          flatScorer);
    }

    private DedupByteVectorValues(
        int size,
        int dimension,
        long dictDataOffset,
        int vectorByteSize,
        int[] ordToDict,
        VectorSimilarityFunction simFunc,
        OrdToDocDISIReaderConfiguration ordToDocConfig,
        IndexInput dataSlice,
        FlatVectorsScorer flatScorer)
        throws IOException {
      this.size = size;
      this.dimension = dimension;
      this.dictDataOffset = dictDataOffset;
      this.vectorByteSize = vectorByteSize;
      this.ordToDict = ordToDict;
      this.simFunc = simFunc;
      this.ordToDocConfig = ordToDocConfig;
      this.dataSlice = dataSlice;
      this.value = new byte[dimension];
      this.flatScorer = flatScorer;
      this.ordToDocReader =
          (!ordToDocConfig.isDense() && !ordToDocConfig.isEmpty())
              ? ordToDocConfig.getDirectMonotonicReader(dataSlice)
              : null;
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
