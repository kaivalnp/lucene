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
package org.apache.lucene.codecs.lucene90;

import static org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat.DIRECT_MONOTONIC_BLOCK_SHIFT;
import static org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat.NUMERIC_BLOCK_SHIFT;
import static org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat.NUMERIC_BLOCK_SIZE;
import static org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat.SKIP_INDEX_LEVEL_SHIFT;
import static org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat.SKIP_INDEX_MAX_LEVEL;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.EmptyDocValuesProducer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.internal.hppc.LongHashSet;
import org.apache.lucene.internal.hppc.LongIntHashMap;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.store.ByteArrayDataOutput;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefBuilder;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.LongsRef;
import org.apache.lucene.util.MathUtil;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.compress.LZ4;
import org.apache.lucene.util.packed.DirectMonotonicWriter;
import org.apache.lucene.util.packed.DirectWriter;

/** writer for {@link Lucene90DocValuesFormat} */
final class Lucene90DocValuesConsumer extends DocValuesConsumer {

  IndexOutput data, meta;
  final int maxDoc;
  private byte[] termsDictBuffer;
  private final int skipIndexIntervalSize;

  /** expert: Creates a new writer */
  public Lucene90DocValuesConsumer(
      SegmentWriteState state,
      int skipIndexIntervalSize,
      String dataCodec,
      String dataExtension,
      String metaCodec,
      String metaExtension)
      throws IOException {
    this.termsDictBuffer = new byte[1 << 14];
    try {
      String dataName =
          IndexFileNames.segmentFileName(
              state.segmentInfo.name, state.segmentSuffix, dataExtension);
      data = state.directory.createOutput(dataName, state.context);
      CodecUtil.writeIndexHeader(
          data,
          dataCodec,
          Lucene90DocValuesFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      String metaName =
          IndexFileNames.segmentFileName(
              state.segmentInfo.name, state.segmentSuffix, metaExtension);
      meta = state.directory.createOutput(metaName, state.context);
      CodecUtil.writeIndexHeader(
          meta,
          metaCodec,
          Lucene90DocValuesFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      maxDoc = state.segmentInfo.maxDoc();
      this.skipIndexIntervalSize = skipIndexIntervalSize;
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, this);
      throw t;
    }
  }

  @Override
  public void close() throws IOException {
    try {
      try {
        if (meta != null) {
          meta.writeInt(-1); // write EOF marker
          CodecUtil.writeFooter(meta); // write checksum
        }
        if (data != null) {
          CodecUtil.writeFooter(data); // write checksum
        }
      } catch (Throwable t) {
        IOUtils.closeWhileSuppressingExceptions(t, data, meta);
        throw t;
      }
      IOUtils.close(data, meta);
    } finally {
      meta = data = null;
    }
  }

  @Override
  public void addNumericField(FieldInfo field, DocValuesProducer valuesProducer)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeByte(Lucene90DocValuesFormat.NUMERIC);
    DocValuesProducer producer =
        new EmptyDocValuesProducer() {
          @Override
          public SortedNumericDocValues getSortedNumeric(FieldInfo field) throws IOException {
            return DocValues.singleton(valuesProducer.getNumeric(field));
          }
        };
    if (field.docValuesSkipIndexType() != DocValuesSkipIndexType.NONE) {
      writeSkipIndex(field, producer);
    }
    writeValues(field, producer, false);
  }

  private static class MinMaxTracker {
    long min, max, numValues, spaceInBits;

    MinMaxTracker() {
      reset();
      spaceInBits = 0;
    }

    private void reset() {
      min = Long.MAX_VALUE;
      max = Long.MIN_VALUE;
      numValues = 0;
    }

    /** Accumulate a new value. */
    void update(long v) {
      min = Math.min(min, v);
      max = Math.max(max, v);
      ++numValues;
    }

    /** Accumulate state from another tracker. */
    void update(MinMaxTracker other) {
      min = Math.min(min, other.min);
      max = Math.max(max, other.max);
      numValues += other.numValues;
    }

    /** Update the required space. */
    void finish() {
      if (max > min) {
        spaceInBits += DirectWriter.unsignedBitsRequired(max - min) * numValues;
      }
    }

    /** Update space usage and get ready for accumulating values for the next block. */
    void nextBlock() {
      finish();
      reset();
    }
  }

  private static class SkipAccumulator {
    int minDocID;
    int maxDocID;
    int docCount;
    long minValue;
    long maxValue;

    SkipAccumulator(int docID) {
      minDocID = docID;
      minValue = Long.MAX_VALUE;
      maxValue = Long.MIN_VALUE;
      docCount = 0;
    }

    boolean isDone(int skipIndexIntervalSize, int valueCount, long nextValue, int nextDoc) {
      if (docCount < skipIndexIntervalSize) {
        return false;
      }
      // Once we reach the interval size, we will keep accepting documents if
      // - next doc value is not a multi-value
      // - current accumulator only contains a single value and next value is the same value
      // - the accumulator is dense and the next doc keeps the density (no gaps)
      return valueCount > 1
          || minValue != maxValue
          || minValue != nextValue
          || docCount != nextDoc - minDocID;
    }

    void accumulate(long value) {
      minValue = Math.min(minValue, value);
      maxValue = Math.max(maxValue, value);
    }

    void accumulate(SkipAccumulator other) {
      assert minDocID <= other.minDocID && maxDocID < other.maxDocID;
      maxDocID = other.maxDocID;
      minValue = Math.min(minValue, other.minValue);
      maxValue = Math.max(maxValue, other.maxValue);
      docCount += other.docCount;
    }

    void nextDoc(int docID) {
      maxDocID = docID;
      ++docCount;
    }

    public static SkipAccumulator merge(List<SkipAccumulator> list, int index, int length) {
      SkipAccumulator acc = new SkipAccumulator(list.get(index).minDocID);
      for (int i = 0; i < length; i++) {
        acc.accumulate(list.get(index + i));
      }
      return acc;
    }
  }

  private void writeSkipIndex(FieldInfo field, DocValuesProducer valuesProducer)
      throws IOException {
    assert field.docValuesSkipIndexType() != DocValuesSkipIndexType.NONE;
    final long start = data.getFilePointer();
    final SortedNumericDocValues values = valuesProducer.getSortedNumeric(field);
    long globalMaxValue = Long.MIN_VALUE;
    long globalMinValue = Long.MAX_VALUE;
    int globalDocCount = 0;
    int maxDocId = -1;
    final List<SkipAccumulator> accumulators = new ArrayList<>();
    SkipAccumulator accumulator = null;
    final int maxAccumulators = 1 << (SKIP_INDEX_LEVEL_SHIFT * (SKIP_INDEX_MAX_LEVEL - 1));
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      final long firstValue = values.nextValue();
      if (accumulator != null
          && accumulator.isDone(skipIndexIntervalSize, values.docValueCount(), firstValue, doc)) {
        globalMaxValue = Math.max(globalMaxValue, accumulator.maxValue);
        globalMinValue = Math.min(globalMinValue, accumulator.minValue);
        globalDocCount += accumulator.docCount;
        maxDocId = accumulator.maxDocID;
        accumulator = null;
        if (accumulators.size() == maxAccumulators) {
          writeLevels(accumulators);
          accumulators.clear();
        }
      }
      if (accumulator == null) {
        accumulator = new SkipAccumulator(doc);
        accumulators.add(accumulator);
      }
      accumulator.nextDoc(doc);
      accumulator.accumulate(firstValue);
      for (int i = 1, end = values.docValueCount(); i < end; ++i) {
        accumulator.accumulate(values.nextValue());
      }
    }

    if (accumulators.isEmpty() == false) {
      globalMaxValue = Math.max(globalMaxValue, accumulator.maxValue);
      globalMinValue = Math.min(globalMinValue, accumulator.minValue);
      globalDocCount += accumulator.docCount;
      maxDocId = accumulator.maxDocID;
      writeLevels(accumulators);
    }
    meta.writeLong(start); // record the start in meta
    meta.writeLong(data.getFilePointer() - start); // record the length
    assert globalDocCount == 0 || globalMaxValue >= globalMinValue;
    meta.writeLong(globalMaxValue);
    meta.writeLong(globalMinValue);
    assert globalDocCount <= maxDocId + 1;
    meta.writeInt(globalDocCount);
    meta.writeInt(maxDocId);
  }

  private void writeLevels(List<SkipAccumulator> accumulators) throws IOException {
    final List<List<SkipAccumulator>> accumulatorsLevels = new ArrayList<>(SKIP_INDEX_MAX_LEVEL);
    accumulatorsLevels.add(accumulators);
    for (int i = 0; i < SKIP_INDEX_MAX_LEVEL - 1; i++) {
      accumulatorsLevels.add(buildLevel(accumulatorsLevels.get(i)));
    }
    int totalAccumulators = accumulators.size();
    for (int index = 0; index < totalAccumulators; index++) {
      // compute how many levels we need to write for the current accumulator
      final int levels = getLevels(index, totalAccumulators);
      // write the number of levels
      data.writeByte((byte) levels);
      // write intervals in reverse order. This is done so we don't
      // need to read all of them in case of slipping
      for (int level = levels - 1; level >= 0; level--) {
        final SkipAccumulator accumulator =
            accumulatorsLevels.get(level).get(index >> (SKIP_INDEX_LEVEL_SHIFT * level));
        data.writeInt(accumulator.maxDocID);
        data.writeInt(accumulator.minDocID);
        data.writeLong(accumulator.maxValue);
        data.writeLong(accumulator.minValue);
        data.writeInt(accumulator.docCount);
      }
    }
  }

  private static List<SkipAccumulator> buildLevel(List<SkipAccumulator> accumulators) {
    final int levelSize = 1 << SKIP_INDEX_LEVEL_SHIFT;
    final List<SkipAccumulator> collector = new ArrayList<>();
    for (int i = 0; i < accumulators.size() - levelSize + 1; i += levelSize) {
      collector.add(SkipAccumulator.merge(accumulators, i, levelSize));
    }
    return collector;
  }

  private static int getLevels(int index, int size) {
    if (Integer.numberOfTrailingZeros(index) >= SKIP_INDEX_LEVEL_SHIFT) {
      // TODO: can we do it in constant time rather than linearly with SKIP_INDEX_MAX_LEVEL?
      final int left = size - index;
      for (int level = SKIP_INDEX_MAX_LEVEL - 1; level > 0; level--) {
        final int numberIntervals = 1 << (SKIP_INDEX_LEVEL_SHIFT * level);
        if (left >= numberIntervals && index % numberIntervals == 0) {
          return level + 1;
        }
      }
    }
    return 1;
  }

  private long[] writeValues(FieldInfo field, DocValuesProducer valuesProducer, boolean ords)
      throws IOException {
    SortedNumericDocValues values = valuesProducer.getSortedNumeric(field);
    final long firstValue;
    if (values.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
      firstValue = values.nextValue();
    } else {
      firstValue = 0L;
    }
    values = valuesProducer.getSortedNumeric(field);
    int numDocsWithValue = 0;
    MinMaxTracker minMax = new MinMaxTracker();
    MinMaxTracker blockMinMax = new MinMaxTracker();
    long gcd = 0;
    LongHashSet uniqueValues = ords ? null : new LongHashSet();
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      for (int i = 0, count = values.docValueCount(); i < count; ++i) {
        long v = values.nextValue();

        if (gcd != 1) {
          if (v < Long.MIN_VALUE / 2 || v > Long.MAX_VALUE / 2) {
            // in that case v - minValue might overflow and make the GCD computation return
            // wrong results. Since these extreme values are unlikely, we just discard
            // GCD computation for them
            gcd = 1;
          } else {
            gcd = MathUtil.gcd(gcd, v - firstValue);
          }
        }

        blockMinMax.update(v);
        if (blockMinMax.numValues == NUMERIC_BLOCK_SIZE) {
          minMax.update(blockMinMax);
          blockMinMax.nextBlock();
        }

        if (uniqueValues != null && uniqueValues.add(v) && uniqueValues.size() > 256) {
          uniqueValues = null;
        }
      }

      numDocsWithValue++;
    }

    minMax.update(blockMinMax);
    minMax.finish();
    blockMinMax.finish();

    if (ords && minMax.numValues > 0) {
      if (minMax.min != 0) {
        throw new IllegalStateException(
            "The min value for ordinals should always be 0, got " + minMax.min);
      }
      if (minMax.max != 0 && gcd != 1) {
        throw new IllegalStateException(
            "GCD compression should never be used on ordinals, found gcd=" + gcd);
      }
    }

    final long numValues = minMax.numValues;
    long min = minMax.min;
    final long max = minMax.max;
    assert blockMinMax.spaceInBits <= minMax.spaceInBits;

    if (numDocsWithValue == 0) { // meta[-2, 0]: No documents with values
      meta.writeLong(-2); // docsWithFieldOffset
      meta.writeLong(0L); // docsWithFieldLength
      meta.writeShort((short) -1); // jumpTableEntryCount
      meta.writeByte((byte) -1); // denseRankPower
    } else if (numDocsWithValue == maxDoc) { // meta[-1, 0]: All documents has values
      meta.writeLong(-1); // docsWithFieldOffset
      meta.writeLong(0L); // docsWithFieldLength
      meta.writeShort((short) -1); // jumpTableEntryCount
      meta.writeByte((byte) -1); // denseRankPower
    } else { // meta[data.offset, data.length]: IndexedDISI structure for documents with values
      long offset = data.getFilePointer();
      meta.writeLong(offset); // docsWithFieldOffset
      values = valuesProducer.getSortedNumeric(field);
      final short jumpTableEntryCount =
          IndexedDISI.writeBitSet(values, data, IndexedDISI.DEFAULT_DENSE_RANK_POWER);
      meta.writeLong(data.getFilePointer() - offset); // docsWithFieldLength
      meta.writeShort(jumpTableEntryCount);
      meta.writeByte(IndexedDISI.DEFAULT_DENSE_RANK_POWER);
    }

    meta.writeLong(numValues);
    final int numBitsPerValue;
    boolean doBlocks = false;
    LongIntHashMap encode = null;
    if (min >= max) { // meta[-1]: All values are 0
      numBitsPerValue = 0;
      meta.writeInt(-1); // tablesize
    } else {
      if (uniqueValues != null
          && uniqueValues.size() > 1
          && DirectWriter.unsignedBitsRequired(uniqueValues.size() - 1)
              < DirectWriter.unsignedBitsRequired((max - min) / gcd)) {
        numBitsPerValue = DirectWriter.unsignedBitsRequired(uniqueValues.size() - 1);
        final long[] sortedUniqueValues = uniqueValues.toArray();
        Arrays.sort(sortedUniqueValues);
        meta.writeInt(sortedUniqueValues.length); // tablesize
        for (long v : sortedUniqueValues) {
          meta.writeLong(v); // table[] entry
        }
        encode = new LongIntHashMap();
        for (int i = 0; i < sortedUniqueValues.length; ++i) {
          encode.put(sortedUniqueValues[i], i);
        }
        min = 0;
        gcd = 1;
      } else {
        uniqueValues = null;
        // we do blocks if that appears to save 10+% storage
        doBlocks =
            minMax.spaceInBits > 0 && (double) blockMinMax.spaceInBits / minMax.spaceInBits <= 0.9;
        if (doBlocks) {
          numBitsPerValue = 0xFF;
          meta.writeInt(-2 - NUMERIC_BLOCK_SHIFT); // tablesize
        } else {
          numBitsPerValue = DirectWriter.unsignedBitsRequired((max - min) / gcd);
          if (gcd == 1
              && min > 0
              && DirectWriter.unsignedBitsRequired(max)
                  == DirectWriter.unsignedBitsRequired(max - min)) {
            min = 0;
          }
          meta.writeInt(-1); // tablesize
        }
      }
    }

    meta.writeByte((byte) numBitsPerValue);
    meta.writeLong(min);
    meta.writeLong(gcd);
    long startOffset = data.getFilePointer();
    meta.writeLong(startOffset); // valueOffset
    long jumpTableOffset = -1;
    if (doBlocks) {
      jumpTableOffset = writeValuesMultipleBlocks(valuesProducer.getSortedNumeric(field), gcd);
    } else if (numBitsPerValue != 0) {
      writeValuesSingleBlock(
          valuesProducer.getSortedNumeric(field), numValues, numBitsPerValue, min, gcd, encode);
    }
    meta.writeLong(data.getFilePointer() - startOffset); // valuesLength
    meta.writeLong(jumpTableOffset);
    return new long[] {numDocsWithValue, numValues};
  }

  private void writeValuesSingleBlock(
      SortedNumericDocValues values,
      long numValues,
      int numBitsPerValue,
      long min,
      long gcd,
      LongIntHashMap encode)
      throws IOException {
    DirectWriter writer = DirectWriter.getInstance(data, numValues, numBitsPerValue);
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      for (int i = 0, count = values.docValueCount(); i < count; ++i) {
        long v = values.nextValue();
        if (encode == null) {
          writer.add((v - min) / gcd);
        } else {
          writer.add(encode.get(v));
        }
      }
    }
    writer.finish();
  }

  // Returns the offset to the jump-table for vBPV
  private long writeValuesMultipleBlocks(SortedNumericDocValues values, long gcd)
      throws IOException {
    long[] offsets = new long[ArrayUtil.oversize(1, Long.BYTES)];
    int offsetsIndex = 0;
    final long[] buffer = new long[NUMERIC_BLOCK_SIZE];
    final ByteBuffersDataOutput encodeBuffer = ByteBuffersDataOutput.newResettableInstance();
    int upTo = 0;
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      for (int i = 0, count = values.docValueCount(); i < count; ++i) {
        buffer[upTo++] = values.nextValue();
        if (upTo == NUMERIC_BLOCK_SIZE) {
          offsets = ArrayUtil.grow(offsets, offsetsIndex + 1);
          offsets[offsetsIndex++] = data.getFilePointer();
          writeBlock(buffer, NUMERIC_BLOCK_SIZE, gcd, encodeBuffer);
          upTo = 0;
        }
      }
    }
    if (upTo > 0) {
      offsets = ArrayUtil.grow(offsets, offsetsIndex + 1);
      offsets[offsetsIndex++] = data.getFilePointer();
      writeBlock(buffer, upTo, gcd, encodeBuffer);
    }

    // All blocks has been written. Flush the offset jump-table
    final long offsetsOrigo = data.getFilePointer();
    for (int i = 0; i < offsetsIndex; i++) {
      data.writeLong(offsets[i]);
    }
    data.writeLong(offsetsOrigo);
    return offsetsOrigo;
  }

  private void writeBlock(long[] values, int length, long gcd, ByteBuffersDataOutput buffer)
      throws IOException {
    assert length > 0;
    long min = values[0];
    long max = values[0];
    for (int i = 1; i < length; ++i) {
      final long v = values[i];
      assert Math.floorMod(values[i] - min, gcd) == 0;
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    if (min == max) {
      data.writeByte((byte) 0);
      data.writeLong(min);
    } else {
      final int bitsPerValue = DirectWriter.unsignedBitsRequired((max - min) / gcd);
      buffer.reset();
      assert buffer.size() == 0;
      final DirectWriter w = DirectWriter.getInstance(buffer, length, bitsPerValue);
      for (int i = 0; i < length; ++i) {
        w.add((values[i] - min) / gcd);
      }
      w.finish();
      data.writeByte((byte) bitsPerValue);
      data.writeLong(min);
      data.writeInt(Math.toIntExact(buffer.size()));
      buffer.copyTo(data);
    }
  }

  @Override
  public void addBinaryField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
    meta.writeInt(field.number);
    meta.writeByte(Lucene90DocValuesFormat.BINARY);

    BinaryDocValues values = valuesProducer.getBinary(field);
    long start = data.getFilePointer();
    meta.writeLong(start); // dataOffset
    int numDocsWithField = 0;
    int minLength = Integer.MAX_VALUE;
    int maxLength = 0;
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      numDocsWithField++;
      BytesRef v = values.binaryValue();
      int length = v.length;
      data.writeBytes(v.bytes, v.offset, v.length);
      minLength = Math.min(length, minLength);
      maxLength = Math.max(length, maxLength);
    }
    assert numDocsWithField <= maxDoc;
    meta.writeLong(data.getFilePointer() - start); // dataLength

    if (numDocsWithField == 0) {
      meta.writeLong(-2); // docsWithFieldOffset
      meta.writeLong(0L); // docsWithFieldLength
      meta.writeShort((short) -1); // jumpTableEntryCount
      meta.writeByte((byte) -1); // denseRankPower
    } else if (numDocsWithField == maxDoc) {
      meta.writeLong(-1); // docsWithFieldOffset
      meta.writeLong(0L); // docsWithFieldLength
      meta.writeShort((short) -1); // jumpTableEntryCount
      meta.writeByte((byte) -1); // denseRankPower
    } else {
      long offset = data.getFilePointer();
      meta.writeLong(offset); // docsWithFieldOffset
      values = valuesProducer.getBinary(field);
      final short jumpTableEntryCount =
          IndexedDISI.writeBitSet(values, data, IndexedDISI.DEFAULT_DENSE_RANK_POWER);
      meta.writeLong(data.getFilePointer() - offset); // docsWithFieldLength
      meta.writeShort(jumpTableEntryCount);
      meta.writeByte(IndexedDISI.DEFAULT_DENSE_RANK_POWER);
    }

    meta.writeInt(numDocsWithField);
    meta.writeInt(minLength);
    meta.writeInt(maxLength);
    if (maxLength > minLength) {
      start = data.getFilePointer();
      meta.writeLong(start);
      meta.writeVInt(DIRECT_MONOTONIC_BLOCK_SHIFT);

      final DirectMonotonicWriter writer =
          DirectMonotonicWriter.getInstance(
              meta, data, numDocsWithField + 1, DIRECT_MONOTONIC_BLOCK_SHIFT);
      long addr = 0;
      writer.add(addr);
      values = valuesProducer.getBinary(field);
      for (int doc = values.nextDoc();
          doc != DocIdSetIterator.NO_MORE_DOCS;
          doc = values.nextDoc()) {
        addr += values.binaryValue().length;
        writer.add(addr);
      }
      writer.finish();
      meta.writeLong(data.getFilePointer() - start);
    }
  }

  @Override
  public void addSortedField(FieldInfo field, DocValuesProducer valuesProducer) throws IOException {
    meta.writeInt(field.number);
    meta.writeByte(Lucene90DocValuesFormat.SORTED);
    doAddSortedField(field, valuesProducer, false);
  }

  private void doAddSortedField(
      FieldInfo field, DocValuesProducer valuesProducer, boolean addTypeByte) throws IOException {
    DocValuesProducer producer =
        new EmptyDocValuesProducer() {
          @Override
          public SortedNumericDocValues getSortedNumeric(FieldInfo field) throws IOException {
            SortedDocValues sorted = valuesProducer.getSorted(field);
            NumericDocValues sortedOrds =
                new NumericDocValues() {
                  @Override
                  public long longValue() throws IOException {
                    return sorted.ordValue();
                  }

                  @Override
                  public boolean advanceExact(int target) throws IOException {
                    return sorted.advanceExact(target);
                  }

                  @Override
                  public int docID() {
                    return sorted.docID();
                  }

                  @Override
                  public int nextDoc() throws IOException {
                    return sorted.nextDoc();
                  }

                  @Override
                  public int advance(int target) throws IOException {
                    return sorted.advance(target);
                  }

                  @Override
                  public long cost() {
                    return sorted.cost();
                  }
                };
            return DocValues.singleton(sortedOrds);
          }
        };
    if (field.docValuesSkipIndexType() != DocValuesSkipIndexType.NONE) {
      writeSkipIndex(field, producer);
    }
    if (addTypeByte) {
      meta.writeByte((byte) 0); // multiValued (0 = singleValued)
    }
    writeValues(field, producer, true);
    addTermsDict(DocValues.singleton(valuesProducer.getSorted(field)));
  }

  private void addTermsDict(SortedSetDocValues values) throws IOException {
    final long size = values.getValueCount();
    meta.writeVLong(size);

    int blockMask = Lucene90DocValuesFormat.TERMS_DICT_BLOCK_LZ4_MASK;
    int shift = Lucene90DocValuesFormat.TERMS_DICT_BLOCK_LZ4_SHIFT;

    meta.writeInt(DIRECT_MONOTONIC_BLOCK_SHIFT);
    ByteBuffersDataOutput addressBuffer = new ByteBuffersDataOutput();
    ByteBuffersIndexOutput addressOutput =
        new ByteBuffersIndexOutput(addressBuffer, "temp", "temp");
    long numBlocks = (size + blockMask) >>> shift;
    DirectMonotonicWriter writer =
        DirectMonotonicWriter.getInstance(
            meta, addressOutput, numBlocks, DIRECT_MONOTONIC_BLOCK_SHIFT);

    BytesRefBuilder previous = new BytesRefBuilder();
    long ord = 0;
    long start = data.getFilePointer();
    int maxLength = 0, maxBlockLength = 0;
    TermsEnum iterator = values.termsEnum();

    LZ4.FastCompressionHashTable ht = new LZ4.FastCompressionHashTable();
    ByteArrayDataOutput bufferedOutput = new ByteArrayDataOutput(termsDictBuffer);
    int dictLength = 0;

    for (BytesRef term = iterator.next(); term != null; term = iterator.next()) {
      if ((ord & blockMask) == 0) {
        if (ord != 0) {
          // flush the previous block
          final int uncompressedLength =
              compressAndGetTermsDictBlockLength(bufferedOutput, dictLength, ht);
          maxBlockLength = Math.max(maxBlockLength, uncompressedLength);
          bufferedOutput.reset(termsDictBuffer);
        }

        writer.add(data.getFilePointer() - start);
        // Write the first term both to the index output, and to the buffer where we'll use it as a
        // dictionary for compression
        data.writeVInt(term.length);
        data.writeBytes(term.bytes, term.offset, term.length);
        bufferedOutput = maybeGrowBuffer(bufferedOutput, term.length);
        bufferedOutput.writeBytes(term.bytes, term.offset, term.length);
        dictLength = term.length;
      } else {
        final int prefixLength = StringHelper.bytesDifference(previous.get(), term);
        final int suffixLength = term.length - prefixLength;
        assert suffixLength > 0; // terms are unique
        // Will write (suffixLength + 1 byte + 2 vint) bytes. Grow the buffer in need.
        bufferedOutput = maybeGrowBuffer(bufferedOutput, suffixLength + 11);
        bufferedOutput.writeByte(
            (byte) (Math.min(prefixLength, 15) | (Math.min(15, suffixLength - 1) << 4)));
        if (prefixLength >= 15) {
          bufferedOutput.writeVInt(prefixLength - 15);
        }
        if (suffixLength >= 16) {
          bufferedOutput.writeVInt(suffixLength - 16);
        }
        bufferedOutput.writeBytes(term.bytes, term.offset + prefixLength, suffixLength);
      }
      maxLength = Math.max(maxLength, term.length);
      previous.copyBytes(term);
      ++ord;
    }
    // Compress and write out the last block
    if (bufferedOutput.getPosition() > dictLength) {
      final int uncompressedLength =
          compressAndGetTermsDictBlockLength(bufferedOutput, dictLength, ht);
      maxBlockLength = Math.max(maxBlockLength, uncompressedLength);
    }

    writer.finish();
    meta.writeInt(maxLength);
    // Write one more int for storing max block length.
    meta.writeInt(maxBlockLength);
    meta.writeLong(start);
    meta.writeLong(data.getFilePointer() - start);
    start = data.getFilePointer();
    addressBuffer.copyTo(data);
    meta.writeLong(start);
    meta.writeLong(data.getFilePointer() - start);

    // Now write the reverse terms index
    writeTermsIndex(values);
  }

  private int compressAndGetTermsDictBlockLength(
      ByteArrayDataOutput bufferedOutput, int dictLength, LZ4.FastCompressionHashTable ht)
      throws IOException {
    int uncompressedLength = bufferedOutput.getPosition() - dictLength;
    data.writeVInt(uncompressedLength);
    LZ4.compressWithDictionary(termsDictBuffer, 0, dictLength, uncompressedLength, data, ht);
    return uncompressedLength;
  }

  private ByteArrayDataOutput maybeGrowBuffer(ByteArrayDataOutput bufferedOutput, int termLength) {
    int pos = bufferedOutput.getPosition(), originalLength = termsDictBuffer.length;
    if (pos + termLength >= originalLength - 1) {
      termsDictBuffer = ArrayUtil.grow(termsDictBuffer, originalLength + termLength);
      bufferedOutput = new ByteArrayDataOutput(termsDictBuffer, pos, termsDictBuffer.length - pos);
    }
    return bufferedOutput;
  }

  private void writeTermsIndex(SortedSetDocValues values) throws IOException {
    final long size = values.getValueCount();
    meta.writeInt(Lucene90DocValuesFormat.TERMS_DICT_REVERSE_INDEX_SHIFT);
    long start = data.getFilePointer();

    long numBlocks =
        1L
            + ((size + Lucene90DocValuesFormat.TERMS_DICT_REVERSE_INDEX_MASK)
                >>> Lucene90DocValuesFormat.TERMS_DICT_REVERSE_INDEX_SHIFT);
    ByteBuffersDataOutput addressBuffer = new ByteBuffersDataOutput();
    DirectMonotonicWriter writer;
    try (ByteBuffersIndexOutput addressOutput =
        new ByteBuffersIndexOutput(addressBuffer, "temp", "temp")) {
      writer =
          DirectMonotonicWriter.getInstance(
              meta, addressOutput, numBlocks, DIRECT_MONOTONIC_BLOCK_SHIFT);
      TermsEnum iterator = values.termsEnum();
      BytesRefBuilder previous = new BytesRefBuilder();
      long offset = 0;
      long ord = 0;
      for (BytesRef term = iterator.next(); term != null; term = iterator.next()) {
        if ((ord & Lucene90DocValuesFormat.TERMS_DICT_REVERSE_INDEX_MASK) == 0) {
          writer.add(offset);
          final int sortKeyLength;
          if (ord == 0) {
            // no previous term: no bytes to write
            sortKeyLength = 0;
          } else {
            sortKeyLength = StringHelper.sortKeyLength(previous.get(), term);
          }
          offset += sortKeyLength;
          data.writeBytes(term.bytes, term.offset, sortKeyLength);
        } else if ((ord & Lucene90DocValuesFormat.TERMS_DICT_REVERSE_INDEX_MASK)
            == Lucene90DocValuesFormat.TERMS_DICT_REVERSE_INDEX_MASK) {
          previous.copyBytes(term);
        }
        ++ord;
      }
      writer.add(offset);
      writer.finish();
      meta.writeLong(start);
      meta.writeLong(data.getFilePointer() - start);
      start = data.getFilePointer();
      addressBuffer.copyTo(data);
      meta.writeLong(start);
      meta.writeLong(data.getFilePointer() - start);
    }
  }

  @Override
  public void addSortedNumericField(FieldInfo field, DocValuesProducer valuesProducer)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeByte(Lucene90DocValuesFormat.SORTED_NUMERIC);
    doAddSortedNumericField(field, valuesProducer, false);
  }

  private void doAddSortedNumericField(
      FieldInfo field, DocValuesProducer valuesProducer, boolean ords) throws IOException {
    if (field.docValuesSkipIndexType() != DocValuesSkipIndexType.NONE) {
      writeSkipIndex(field, valuesProducer);
    }
    if (ords) {
      meta.writeByte((byte) 1); // multiValued (1 = multiValued)
    }
    long[] stats = writeValues(field, valuesProducer, ords);
    int numDocsWithField = Math.toIntExact(stats[0]);
    long numValues = stats[1];
    assert numValues >= numDocsWithField;

    meta.writeInt(numDocsWithField);
    if (numValues > numDocsWithField) {
      long start = data.getFilePointer();
      meta.writeLong(start);
      meta.writeVInt(DIRECT_MONOTONIC_BLOCK_SHIFT);

      final DirectMonotonicWriter addressesWriter =
          DirectMonotonicWriter.getInstance(
              meta, data, numDocsWithField + 1L, DIRECT_MONOTONIC_BLOCK_SHIFT);
      long addr = 0;
      addressesWriter.add(addr);
      SortedNumericDocValues values = valuesProducer.getSortedNumeric(field);
      for (int doc = values.nextDoc();
          doc != DocIdSetIterator.NO_MORE_DOCS;
          doc = values.nextDoc()) {
        addr += values.docValueCount();
        addressesWriter.add(addr);
      }
      addressesWriter.finish();
      meta.writeLong(data.getFilePointer() - start);
    }
  }

  private static boolean isSingleValued(SortedSetDocValues values) throws IOException {
    if (DocValues.unwrapSingleton(values) != null) {
      return true;
    }

    assert values.docID() == -1;
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      int docValueCount = values.docValueCount();
      assert docValueCount > 0;
      if (docValueCount > 1) {
        return false;
      }
    }
    return true;
  }

  @Override
  public void addSortedSetField(FieldInfo field, DocValuesProducer valuesProducer)
      throws IOException {
    meta.writeInt(field.number);
    meta.writeByte(Lucene90DocValuesFormat.SORTED_SET);

    if (isSingleValued(valuesProducer.getSortedSet(field))) {

      doAddSortedField(
          field,
          new EmptyDocValuesProducer() {
            @Override
            public SortedDocValues getSorted(FieldInfo field) throws IOException {
              return SortedSetSelector.wrap(
                  valuesProducer.getSortedSet(field), SortedSetSelector.Type.MIN);
            }
          },
          true);
      return;
    }

    doAddSortedNumericField(
        field,
        new EmptyDocValuesProducer() {
          @Override
          public SortedNumericDocValues getSortedNumeric(FieldInfo field) throws IOException {
            SortedSetDocValues values = valuesProducer.getSortedSet(field);
            return new SortedNumericDocValues() {

              long[] ords = LongsRef.EMPTY_LONGS;
              int i, docValueCount;

              @Override
              public long nextValue() throws IOException {
                return ords[i++];
              }

              @Override
              public int docValueCount() {
                return docValueCount;
              }

              @Override
              public boolean advanceExact(int target) throws IOException {
                throw new UnsupportedOperationException();
              }

              @Override
              public int docID() {
                return values.docID();
              }

              @Override
              public int nextDoc() throws IOException {
                int doc = values.nextDoc();
                if (doc != NO_MORE_DOCS) {
                  docValueCount = values.docValueCount();
                  ords = ArrayUtil.grow(ords, docValueCount);
                  for (int j = 0; j < docValueCount; j++) {
                    ords[j] = values.nextOrd();
                  }
                  i = 0;
                }
                return doc;
              }

              @Override
              public int advance(int target) throws IOException {
                throw new UnsupportedOperationException();
              }

              @Override
              public long cost() {
                return values.cost();
              }
            };
          }
        },
        true);

    addTermsDict(valuesProducer.getSortedSet(field));
  }
}
