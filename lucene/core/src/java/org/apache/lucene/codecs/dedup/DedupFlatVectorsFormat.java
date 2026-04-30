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
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * A {@link FlatVectorsFormat} that de-duplicates stored vectors across all fields in a segment.
 * Unique vectors are stored once in a shared dictionary (.dvd), and each field maintains a compact
 * ordinal mapping from its document ords to dictionary ords (.dvm/.dvd).
 *
 * <h2>.dvd (dedup vector data) file</h2>
 *
 * <ul>
 *   <li>Shared dictionary: unique vectors stored contiguously, accessible by dictOrd
 *   <li>Per-field ord-to-dictOrd mappings (vint encoded)
 *   <li>Per-field OrdToDoc DISI data (for sparse fields)
 * </ul>
 *
 * <h2>.dvm (dedup vector meta) file</h2>
 *
 * <ul>
 *   <li>Dictionary size (number of unique vectors), dimension, vector encoding
 *   <li>Per-field: field number, size, ordToDoc config, map offset/length
 *   <li>-1 sentinel marking end of fields
 * </ul>
 *
 * <p>The writer performs no temporary-file IO. Dictionary bytes stream directly into the main
 * {@code .dvd} output. Hash-collision verification uses already-resident sources: on-heap typed
 * arrays during flush, and already-open {@link org.apache.lucene.codecs.KnnVectorsReader}s from
 * {@link org.apache.lucene.index.MergeState} during merge.
 *
 * <p>When a source segment being merged was itself written with this format, the merge path skips
 * per-doc hashing through a lazy {@code sourceDictOrd → targetDictOrd} cache and a same-sub int-ord
 * shortcut on hash collision. This also drops source dict entries whose referencing docs have all
 * been deleted — their dict ords are never touched, so dead vectors do not accumulate across merge
 * cycles.
 *
 * <p>See {@code lucene/dev-docs/dedup-vectors-format-design.md} for the full design.
 *
 * @lucene.experimental
 */
public final class DedupFlatVectorsFormat extends FlatVectorsFormat {

  public static final String NAME = "DedupFlatVectorsFormat";

  static final String META_CODEC_NAME = "DedupFlatVectorsFormatMeta";
  static final String DATA_CODEC_NAME = "DedupFlatVectorsFormatData";
  static final String META_EXTENSION = "dvm";
  static final String DATA_EXTENSION = "dvd";

  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;

  private final FlatVectorsScorer vectorsScorer;

  /** Creates a dedup format with the default scorer. */
  public DedupFlatVectorsFormat() {
    this(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
  }

  /** Creates a dedup format with the given scorer. */
  public DedupFlatVectorsFormat(FlatVectorsScorer vectorsScorer) {
    super(NAME);
    this.vectorsScorer = vectorsScorer;
  }

  @Override
  public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new DedupFlatVectorsWriter(state, vectorsScorer);
  }

  @Override
  public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new DedupFlatVectorsReader(state, vectorsScorer);
  }

  @Override
  public String toString() {
    return "DedupFlatVectorsFormat(vectorsScorer=" + vectorsScorer + ")";
  }
}
