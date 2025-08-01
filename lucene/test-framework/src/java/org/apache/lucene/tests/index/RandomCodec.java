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
package org.apache.lucene.tests.index;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.PointsFormat;
import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.codecs.PointsWriter;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.blocktreeords.BlockTreeOrdsPostingsFormat;
import org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat;
import org.apache.lucene.codecs.lucene90.Lucene90PointsReader;
import org.apache.lucene.codecs.lucene90.Lucene90PointsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.memory.DirectPostingsFormat;
import org.apache.lucene.codecs.memory.FSTPostingsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.PointValues.IntersectVisitor;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.codecs.asserting.AssertingCodec;
import org.apache.lucene.tests.codecs.asserting.AssertingDocValuesFormat;
import org.apache.lucene.tests.codecs.asserting.AssertingKnnVectorsFormat;
import org.apache.lucene.tests.codecs.asserting.AssertingPointsFormat;
import org.apache.lucene.tests.codecs.asserting.AssertingPostingsFormat;
import org.apache.lucene.tests.codecs.blockterms.LuceneFixedGap;
import org.apache.lucene.tests.codecs.blockterms.LuceneVarGapDocFreqInterval;
import org.apache.lucene.tests.codecs.blockterms.LuceneVarGapFixedInterval;
import org.apache.lucene.tests.codecs.bloom.TestBloomFilteredLucenePostings;
import org.apache.lucene.tests.codecs.mockrandom.MockRandomPostingsFormat;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.IORunnable;
import org.apache.lucene.util.bkd.BKDConfig;
import org.apache.lucene.util.bkd.BKDWriter;

/**
 * Codec that assigns per-field random postings formats.
 *
 * <p>The same field/format assignment will happen regardless of order, a hash is computed up front
 * that determines the mapping. This means fields can be put into things like HashSets and added to
 * documents in different orders and the test will still be deterministic and reproducable.
 */
public class RandomCodec extends AssertingCodec {
  /** Shuffled list of postings formats to use for new mappings */
  private List<PostingsFormat> formats = new ArrayList<>();

  /** Shuffled list of docvalues formats to use for new mappings */
  private List<DocValuesFormat> dvFormats = new ArrayList<>();

  /** Shuffled list of knn formats to use for new mappings */
  private List<KnnVectorsFormat> knnFormats = new ArrayList<>();

  /** unique set of format names this codec knows about */
  public Set<String> formatNames = new HashSet<>();

  /** unique set of docvalues format names this codec knows about */
  public Set<String> dvFormatNames = new HashSet<>();

  /** unique set of knn format names this codec knows about */
  public Set<String> knnFormatNames = new HashSet<>();

  public final Set<String> avoidCodecs;

  /** memorized field to postingsformat mappings */
  // note: we have to sync this map even though it's just for debugging/toString,
  // otherwise DWPT's .toString() calls that iterate over the map can
  // cause concurrentmodificationexception if indexwriter's infostream is on
  private Map<String, PostingsFormat> previousMappings =
      Collections.synchronizedMap(new HashMap<String, PostingsFormat>());

  private Map<String, DocValuesFormat> previousDVMappings =
      Collections.synchronizedMap(new HashMap<String, DocValuesFormat>());

  private Map<String, KnnVectorsFormat> previousKnnMappings =
      Collections.synchronizedMap(new HashMap<String, KnnVectorsFormat>());

  private final int perFieldSeed;

  // a little messy: randomize the default codec's parameters here.
  // with the default values, we have e,g, 512 points in leaf nodes,
  // which is less effective for testing.
  // TODO: improve how we randomize this...
  private final int maxPointsInLeafNode;
  private final double maxMBSortInHeap;
  private final int bkdSplitRandomSeed;

  @Override
  public PointsFormat pointsFormat() {
    return new AssertingPointsFormat(
        new PointsFormat() {
          @Override
          public PointsWriter fieldsWriter(SegmentWriteState writeState) throws IOException {

            // Randomize how BKDWriter chooses its splits:

            return new Lucene90PointsWriter(writeState, maxPointsInLeafNode, maxMBSortInHeap) {
              @Override
              public void writeField(FieldInfo fieldInfo, PointsReader reader) throws IOException {

                PointValues.PointTree values = reader.getValues(fieldInfo.name).getPointTree();

                BKDConfig config =
                    new BKDConfig(
                        fieldInfo.getPointDimensionCount(),
                        fieldInfo.getPointIndexDimensionCount(),
                        fieldInfo.getPointNumBytes(),
                        maxPointsInLeafNode);

                try (BKDWriter writer =
                    new RandomlySplittingBKDWriter(
                        writeState.segmentInfo.maxDoc(),
                        writeState.directory,
                        writeState.segmentInfo.name,
                        config,
                        maxMBSortInHeap,
                        values.size(),
                        bkdSplitRandomSeed ^ fieldInfo.name.hashCode())) {
                  values.visitDocValues(
                      new IntersectVisitor() {
                        @Override
                        public void visit(int docID) {
                          throw new IllegalStateException();
                        }

                        @Override
                        public void visit(int docID, byte[] packedValue) throws IOException {
                          writer.add(packedValue, docID);
                        }

                        @Override
                        public PointValues.Relation compare(
                            byte[] minPackedValue, byte[] maxPackedValue) {
                          return PointValues.Relation.CELL_CROSSES_QUERY;
                        }
                      });

                  // We could have 0 points on merge since all docs with dimensional fields may be
                  // deleted:
                  IORunnable finalizer = writer.finish(metaOut, indexOut, dataOut);
                  if (finalizer != null) {
                    metaOut.writeInt(fieldInfo.number);
                    finalizer.run();
                  }
                }
              }
            };
          }

          @Override
          public PointsReader fieldsReader(SegmentReadState readState) throws IOException {
            return new Lucene90PointsReader(readState);
          }
        });
  }

  @Override
  public PostingsFormat getPostingsFormatForField(String name) {
    PostingsFormat codec = previousMappings.get(name);
    if (codec == null) {
      codec = formats.get(Math.abs(perFieldSeed ^ name.hashCode()) % formats.size());
      previousMappings.put(name, codec);
      // Safety:
      assert previousMappings.size() < 10000 : "test went insane";
    }
    return codec;
  }

  @Override
  public DocValuesFormat getDocValuesFormatForField(String name) {
    DocValuesFormat codec = previousDVMappings.get(name);
    if (codec == null) {
      codec = dvFormats.get(Math.abs(perFieldSeed ^ name.hashCode()) % dvFormats.size());
      previousDVMappings.put(name, codec);
      // Safety:
      assert previousDVMappings.size() < 10000 : "test went insane";
    }
    return codec;
  }

  @Override
  public KnnVectorsFormat getKnnVectorsFormatForField(String name) {
    KnnVectorsFormat format = previousKnnMappings.get(name);
    if (format == null) {
      format = knnFormats.get(Math.abs(perFieldSeed ^ name.hashCode()) % knnFormats.size());
      previousKnnMappings.put(name, format);
      // Safety:
      assert previousKnnMappings.size() < 10000 : "test went insane";
    }
    return format;
  }

  public RandomCodec(Random random, Set<String> avoidCodecs) {
    this.perFieldSeed = random.nextInt();
    this.avoidCodecs = avoidCodecs;
    // TODO: make it possible to specify min/max iterms per
    // block via CL:
    int minItemsPerBlock = TestUtil.nextInt(random, 2, 100);
    int maxItemsPerBlock = 2 * (Math.max(2, minItemsPerBlock - 1)) + random.nextInt(100);
    int lowFreqCutoff = TestUtil.nextInt(random, 2, 100);

    maxPointsInLeafNode = TestUtil.nextInt(random, 16, 2048);
    maxMBSortInHeap = 5.0 + (3 * random.nextDouble());
    bkdSplitRandomSeed = random.nextInt();

    add(
        avoidCodecs,
        TestUtil.getDefaultPostingsFormat(minItemsPerBlock, maxItemsPerBlock),
        new FSTPostingsFormat(),
        new DirectPostingsFormat(
            LuceneTestCase.rarely(random)
                ? 1
                : (LuceneTestCase.rarely(random) ? Integer.MAX_VALUE : maxItemsPerBlock),
            LuceneTestCase.rarely(random)
                ? 1
                : (LuceneTestCase.rarely(random) ? Integer.MAX_VALUE : lowFreqCutoff)),
        // TODO as a PostingsFormat which wraps others, we should allow
        // TestBloomFilteredLucenePostings to be constructed
        // with a choice of concrete PostingsFormats. Maybe useful to have a generic means of
        // marking and dealing
        // with such "wrapper" classes?
        new TestBloomFilteredLucenePostings(),
        new MockRandomPostingsFormat(random),
        new BlockTreeOrdsPostingsFormat(minItemsPerBlock, maxItemsPerBlock),
        new LuceneFixedGap(TestUtil.nextInt(random, 1, 1000)),
        new LuceneVarGapFixedInterval(TestUtil.nextInt(random, 1, 1000)),
        new LuceneVarGapDocFreqInterval(
            TestUtil.nextInt(random, 1, 100), TestUtil.nextInt(random, 1, 1000)),
        TestUtil.getDefaultPostingsFormat(),
        new AssertingPostingsFormat());

    addDocValues(
        avoidCodecs,
        TestUtil.getDefaultDocValuesFormat(),
        new Lucene90DocValuesFormat(),
        new AssertingDocValuesFormat());

    boolean concurrentKnnMerging = random.nextBoolean();
    addKnn(
        avoidCodecs,
        TestUtil.getDefaultKnnVectorsFormat(),
        new Lucene99HnswVectorsFormat(
            TestUtil.nextInt(random, 5, 50),
            TestUtil.nextInt(random, 10, 50),
            concurrentKnnMerging ? TestUtil.nextInt(random, 2, 8) : 1,
            concurrentKnnMerging ? ForkJoinPool.commonPool() : null),
        new Lucene99HnswScalarQuantizedVectorsFormat(
            TestUtil.nextInt(random, 5, 50),
            TestUtil.nextInt(random, 10, 50),
            concurrentKnnMerging ? TestUtil.nextInt(random, 2, 8) : 1,
            7,
            false,
            randomConfidenceInterval(random),
            concurrentKnnMerging ? ForkJoinPool.commonPool() : null),
        // TODO: also test 4-bit quantization, but this must somehow be restricted to even-length
        // fields
        /*
         * new Lucene99HnswScalarQuantizedVectorsFormat(TestUtil.nextInt(random, 5, 50),
         *                                              TestUtil.nextInt(random, 10, 50),
         *                                              1,
         *                                              4,
         *                                              random.nextBoolean(),
         *                                              randomConfidenceInterval(random),
         *                                              null),
         * new Lucene99HnswScalarQuantizedVectorsFormat(TestUtil.nextInt(random, 5, 50),
         *                                              TestUtil.nextInt(random, 10, 50),
         *                                              TestUtil.nextInt(random, 2, 8),
         *                                              4,
         *                                              random.nextBoolean(),
         *                                              randomConfidenceInterval(random),
         *                                              ForkJoinPool.commonPool()),
         */
        new AssertingKnnVectorsFormat());

    Collections.shuffle(formats, random);
    Collections.shuffle(dvFormats, random);
    Collections.shuffle(knnFormats, random);

    // Avoid too many open files:
    if (formats.size() > 4) {
      formats = formats.subList(0, 4);
    }
    if (dvFormats.size() > 4) {
      dvFormats = dvFormats.subList(0, 4);
    }
    if (knnFormats.size() > 4) {
      knnFormats = knnFormats.subList(0, 4);
    }
  }

  private final Float randomConfidenceInterval(Random random) {
    switch (random.nextInt(3)) {
      default:
      case 0:
        return null;
      case 1:
        return 0f;
      case 2:
        return random.nextFloat(0.9f, 1f);
    }
  }

  public RandomCodec(Random random) {
    this(random, Collections.<String>emptySet());
  }

  private final void add(Set<String> avoidCodecs, PostingsFormat... postings) {
    for (PostingsFormat p : postings) {
      if (!avoidCodecs.contains(p.getName())) {
        formats.add(p);
        formatNames.add(p.getName());
      }
    }
  }

  private final void addDocValues(Set<String> avoidCodecs, DocValuesFormat... docvalues) {
    for (DocValuesFormat d : docvalues) {
      if (!avoidCodecs.contains(d.getName())) {
        dvFormats.add(d);
        dvFormatNames.add(d.getName());
      }
    }
  }

  private final void addKnn(Set<String> avoidCodecs, KnnVectorsFormat... knnFormat) {
    for (KnnVectorsFormat kf : knnFormat) {
      if (!avoidCodecs.contains(kf.getName())) {
        knnFormats.add(kf);
        knnFormatNames.add(kf.getName());
      }
    }
  }

  @Override
  public String toString() {
    return super.toString()
        + ": "
        + previousMappings.toString()
        + ", knn_vectors:"
        + previousKnnMappings.toString()
        + ", docValues:"
        + previousDVMappings.toString()
        + ", maxPointsInLeafNode="
        + maxPointsInLeafNode
        + ", maxMBSortInHeap="
        + maxMBSortInHeap;
  }

  /**
   * Just like {@link BKDWriter} except it evilly picks random ways to split cells on recursion to
   * try to provoke geo APIs that get upset at fun rectangles.
   */
  private static class RandomlySplittingBKDWriter extends BKDWriter {

    final Random random;

    public RandomlySplittingBKDWriter(
        int maxDoc,
        Directory tempDir,
        String tempFileNamePrefix,
        BKDConfig config,
        double maxMBSortInHeap,
        long totalPointCount,
        int randomSeed)
        throws IOException {
      super(maxDoc, tempDir, tempFileNamePrefix, config, maxMBSortInHeap, totalPointCount);
      this.random = new Random(randomSeed);
    }

    @Override
    protected int split(byte[] minPackedValue, byte[] maxPackedValue, int[] parentDims) {
      // BKD normally defaults by the widest dimension, to try to make as squarish cells as
      // possible, but we just pick a random one ;)
      return random.nextInt(config.numIndexDims());
    }
  }
}
