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

package org.apache.lucene.codecs.perfield;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.internal.hppc.ObjectCursor;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.HnswGraph;

/**
 * Enables per field numeric vector support.
 *
 * <p>Note, when extending this class, the name ({@link #getName}) is written into the index. In
 * order for the field to be read, the name must resolve to your implementation via {@link
 * #forName(String)}. This method uses Java's {@link ServiceLoader Service Provider Interface} to
 * resolve format names.
 *
 * <p>Files written by each numeric vectors format have an additional suffix containing the format
 * name. For example, in a per-field configuration instead of <code>_1.dat</code> filenames would
 * look like <code>_1_Lucene40_0.dat</code>.
 *
 * @see ServiceLoader
 * @lucene.experimental
 */
public abstract class PerFieldKnnVectorsFormat extends KnnVectorsFormat {
  /** Name of this {@link KnnVectorsFormat}. */
  public static final String PER_FIELD_NAME = "PerFieldVectors90";

  /** {@link FieldInfo} attribute name used to store the format name for each field. */
  public static final String PER_FIELD_FORMAT_KEY =
      PerFieldKnnVectorsFormat.class.getSimpleName() + ".format";

  /** {@link FieldInfo} attribute name used to store the segment suffix name for each field. */
  public static final String PER_FIELD_SUFFIX_KEY =
      PerFieldKnnVectorsFormat.class.getSimpleName() + ".suffix";

  /** Sole constructor. */
  protected PerFieldKnnVectorsFormat() {
    super(PER_FIELD_NAME);
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new FieldsWriter(state);
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new FieldsReader(state);
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return getKnnVectorsFormatForField(fieldName).getMaxDimensions(fieldName);
  }

  /**
   * Returns the numeric vector format that should be used for writing new segments of <code>field
   * </code>.
   *
   * <p>The field to format mapping is written to the index, so this method is only invoked when
   * writing, not when reading.
   */
  public abstract KnnVectorsFormat getKnnVectorsFormatForField(String field);

  private class FieldsWriter extends KnnVectorsWriter {
    private final Map<KnnVectorsFormat, WriterAndSuffix> formats;
    private final Map<String, Integer> suffixes = new HashMap<>();
    private final SegmentWriteState segmentWriteState;

    FieldsWriter(SegmentWriteState segmentWriteState) {
      this.segmentWriteState = segmentWriteState;
      formats = new HashMap<>();
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
      KnnVectorsWriter writer = getInstance(fieldInfo);
      return writer.addField(fieldInfo);
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
      for (WriterAndSuffix was : formats.values()) {
        was.writer.flush(maxDoc, sortMap);
      }
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
      getInstance(fieldInfo).mergeOneField(fieldInfo, mergeState);
    }

    @Override
    public void finish() throws IOException {
      for (WriterAndSuffix was : formats.values()) {
        was.writer.finish();
      }
    }

    @Override
    public void close() throws IOException {
      IOUtils.close(formats.values());
    }

    private KnnVectorsWriter getInstance(FieldInfo field) throws IOException {
      KnnVectorsFormat format = getKnnVectorsFormatForField(field.name);
      if (format == null) {
        throw new IllegalStateException(
            "invalid null KnnVectorsFormat for field=\"" + field.name + "\"");
      }
      final String formatName = format.getName();

      field.putAttribute(PER_FIELD_FORMAT_KEY, formatName);
      Integer suffix;

      WriterAndSuffix writerAndSuffix = formats.get(format);
      if (writerAndSuffix == null) {
        // First time we are seeing this format; create a new instance

        suffix = suffixes.get(formatName);
        if (suffix == null) {
          suffix = 0;
        } else {
          suffix = suffix + 1;
        }
        suffixes.put(formatName, suffix);

        String segmentSuffix =
            getFullSegmentSuffix(
                segmentWriteState.segmentSuffix, getSuffix(formatName, Integer.toString(suffix)));
        writerAndSuffix =
            new WriterAndSuffix(
                format.fieldsWriter(new SegmentWriteState(segmentWriteState, segmentSuffix)),
                suffix);
        formats.put(format, writerAndSuffix);
      } else {
        // we've already seen this format, so just grab its suffix
        assert suffixes.containsKey(formatName);
        suffix = writerAndSuffix.suffix;
      }
      field.putAttribute(PER_FIELD_SUFFIX_KEY, Integer.toString(suffix));
      return writerAndSuffix.writer;
    }

    @Override
    public long ramBytesUsed() {
      long total = 0;
      for (WriterAndSuffix was : formats.values()) {
        total += was.writer.ramBytesUsed();
      }
      return total;
    }
  }

  /** VectorReader that can wrap multiple delegate readers, selected by field. */
  public static class FieldsReader extends KnnVectorsReader implements HnswGraphProvider {

    private final IntObjectHashMap<KnnVectorsReader> fields = new IntObjectHashMap<>();
    private final FieldInfos fieldInfos;

    /**
     * Create a FieldsReader over a segment, opening VectorReaders for each KnnVectorsFormat
     * specified by the indexed numeric vector fields.
     *
     * @param readState defines the fields
     * @throws IOException if one of the delegate readers throws
     */
    public FieldsReader(final SegmentReadState readState) throws IOException {
      this.fieldInfos = readState.fieldInfos;
      // Init each unique format:
      Map<String, KnnVectorsReader> formats = new HashMap<>();
      try {
        // Read field name -> format name
        for (FieldInfo fi : readState.fieldInfos) {
          if (fi.hasVectorValues()) {
            final String fieldName = fi.name;
            final String formatName = fi.getAttribute(PER_FIELD_FORMAT_KEY);
            if (formatName != null) {
              // null formatName means the field is in fieldInfos, but has no vectors!
              final String suffix = fi.getAttribute(PER_FIELD_SUFFIX_KEY);
              if (suffix == null) {
                throw new IllegalStateException(
                    "missing attribute: " + PER_FIELD_SUFFIX_KEY + " for field: " + fieldName);
              }
              KnnVectorsFormat format = KnnVectorsFormat.forName(formatName);
              String segmentSuffix =
                  getFullSegmentSuffix(readState.segmentSuffix, getSuffix(formatName, suffix));
              if (!formats.containsKey(segmentSuffix)) {
                formats.put(
                    segmentSuffix,
                    format.fieldsReader(new SegmentReadState(readState, segmentSuffix)));
              }
              fields.put(fi.number, formats.get(segmentSuffix));
            }
          }
        }
      } catch (Throwable t) {
        IOUtils.closeWhileSuppressingExceptions(t, formats.values());
        throw t;
      }
    }

    private FieldsReader(final FieldsReader fieldsReader) {
      this.fieldInfos = fieldsReader.fieldInfos;
      for (FieldInfo fi : this.fieldInfos) {
        if (fi.hasVectorValues() && fieldsReader.fields.containsKey(fi.number)) {
          this.fields.put(fi.number, fieldsReader.fields.get(fi.number).getMergeInstance());
        }
      }
    }

    @Override
    public KnnVectorsReader getMergeInstance() {
      return new FieldsReader(this);
    }

    @Override
    public void finishMerge() throws IOException {
      for (ObjectCursor<KnnVectorsReader> knnVectorReader : fields.values()) {
        knnVectorReader.value.finishMerge();
      }
    }

    /**
     * Return the underlying VectorReader for the given field
     *
     * @param field the name of a numeric vector field
     */
    public KnnVectorsReader getFieldReader(String field) {
      final FieldInfo info = fieldInfos.fieldInfo(field);
      if (info == null) {
        return null;
      }
      return fields.get(info.number);
    }

    @Override
    public void checkIntegrity() throws IOException {
      for (ObjectCursor<KnnVectorsReader> cursor : fields.values()) {
        cursor.value.checkIntegrity();
      }
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
      final FieldInfo info = fieldInfos.fieldInfo(field);
      final KnnVectorsReader reader;
      if (info == null || (reader = fields.get(info.number)) == null) {
        return null;
      }
      return reader.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
      final FieldInfo info = fieldInfos.fieldInfo(field);
      final KnnVectorsReader reader;
      if (info == null || (reader = fields.get(info.number)) == null) {
        return null;
      }
      return reader.getByteVectorValues(field);
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
        throws IOException {
      final FieldInfo info = fieldInfos.fieldInfo(field);
      final KnnVectorsReader reader;
      if (info == null || (reader = fields.get(info.number)) == null) {
        return;
      }
      reader.search(field, target, knnCollector, acceptDocs);
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
        throws IOException {
      final FieldInfo info = fieldInfos.fieldInfo(field);
      final KnnVectorsReader reader;
      if (info == null || (reader = fields.get(info.number)) == null) {
        return;
      }
      reader.search(field, target, knnCollector, acceptDocs);
    }

    @Override
    public HnswGraph getGraph(String field) throws IOException {
      final FieldInfo info = fieldInfos.fieldInfo(field);
      KnnVectorsReader knnVectorsReader = fields.get(info.number);
      if (knnVectorsReader instanceof HnswGraphProvider) {
        return ((HnswGraphProvider) knnVectorsReader).getGraph(field);
      } else {
        return null;
      }
    }

    @Override
    public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
      KnnVectorsReader knnVectorsReader = fields.get(fieldInfo.number);
      return knnVectorsReader.getOffHeapByteSize(fieldInfo);
    }

    @Override
    public void close() throws IOException {
      List<KnnVectorsReader> readers = new ArrayList<>(fields.size());
      for (ObjectCursor<KnnVectorsReader> cursor : fields.values()) {
        readers.add(cursor.value);
      }
      IOUtils.close(readers);
    }
  }

  static String getSuffix(String formatName, String suffix) {
    return formatName + "_" + suffix;
  }

  static String getFullSegmentSuffix(String outerSegmentSuffix, String segmentSuffix) {
    if (outerSegmentSuffix.length() == 0) {
      return segmentSuffix;
    } else {
      return outerSegmentSuffix + "_" + segmentSuffix;
    }
  }

  private record WriterAndSuffix(KnnVectorsWriter writer, int suffix) implements Closeable {

    @Override
    public void close() throws IOException {
      writer.close();
    }
  }
}
