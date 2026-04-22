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
import java.util.Arrays;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;

/** Tests for {@link DedupFlatVectorsFormat}. */
public class TestDedupFlatVectorsFormat extends LuceneTestCase {

  private static IndexWriterConfig iwc() {
    return new IndexWriterConfig()
        .setCodec(TestUtil.alwaysKnnVectorsFormat(new DedupFlatVectorsFormat()));
  }

  /** Verify that duplicate float vectors within a single field are read back correctly. */
  public void testSingleFieldDuplicateFloats() throws IOException {
    float[] shared = {1f, 2f, 3f};
    float[] unique = {4f, 5f, 6f};

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, iwc())) {
      for (int i = 0; i < 5; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("f", shared, VectorSimilarityFunction.EUCLIDEAN));
        w.addDocument(doc);
      }
      Document doc = new Document();
      doc.add(new KnnFloatVectorField("f", unique, VectorSimilarityFunction.EUCLIDEAN));
      w.addDocument(doc);
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        FloatVectorValues values = leaf.getFloatVectorValues("f");
        assertEquals(6, values.size());

        KnnVectorValues.DocIndexIterator iter = values.iterator();
        for (int i = 0; i < 5; i++) {
          int docId = iter.nextDoc();
          assertNotEquals(NO_MORE_DOCS, docId);
          assertArrayEquals(shared, values.vectorValue(iter.index()), 0f);
        }
        int docId = iter.nextDoc();
        assertNotEquals(NO_MORE_DOCS, docId);
        assertArrayEquals(unique, values.vectorValue(iter.index()), 0f);
        assertEquals(NO_MORE_DOCS, iter.nextDoc());
      }
    }
  }

  /** Verify that duplicate byte vectors within a single field are read back correctly. */
  public void testSingleFieldDuplicateBytes() throws IOException {
    byte[] shared = {1, 2, 3};
    byte[] unique = {4, 5, 6};

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, iwc())) {
      for (int i = 0; i < 5; i++) {
        Document doc = new Document();
        doc.add(new KnnByteVectorField("f", shared, VectorSimilarityFunction.EUCLIDEAN));
        w.addDocument(doc);
      }
      Document doc = new Document();
      doc.add(new KnnByteVectorField("f", unique, VectorSimilarityFunction.EUCLIDEAN));
      w.addDocument(doc);
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        var values = leaf.getByteVectorValues("f");
        assertEquals(6, values.size());

        KnnVectorValues.DocIndexIterator iter = values.iterator();
        for (int i = 0; i < 5; i++) {
          iter.nextDoc();
          assertArrayEquals(shared, values.vectorValue(iter.index()));
        }
        iter.nextDoc();
        assertArrayEquals(unique, values.vectorValue(iter.index()));
      }
    }
  }

  /**
   * Verify cross-field dedup: two fields with the same vector should result in less disk usage than
   * storing each independently.
   */
  public void testCrossFieldDedupDiskSavings() throws IOException {
    int dim = 128;
    float[] sharedVec = new float[dim];
    Arrays.fill(sharedVec, 0.5f);

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, iwc())) {
      for (int i = 0; i < 100; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("field_a", sharedVec, VectorSimilarityFunction.EUCLIDEAN));
        doc.add(new KnnFloatVectorField("field_b", sharedVec, VectorSimilarityFunction.EUCLIDEAN));
        w.addDocument(doc);
      }
      w.forceMerge(1);
      w.commit();

      // The .dvd file should store the vector only once (1 unique vector),
      // not 200 times (100 docs × 2 fields).
      long dvdSize = 0;
      for (String file : dir.listAll()) {
        if (file.endsWith(DedupFlatVectorsFormat.DATA_EXTENSION)) {
          dvdSize += dir.fileLength(file);
        }
      }

      // Without dedup: 200 vectors × 128 dims × 4 bytes = 102,400 bytes
      // With dedup: 1 unique vector × 128 dims × 4 bytes = 512 bytes (plus overhead)
      long noDedupSize = 200L * dim * Float.BYTES;
      assertTrue(
          "dvd file (" + dvdSize + ") should be much smaller than no-dedup (" + noDedupSize + ")",
          dvdSize < noDedupSize / 2);
    }
  }

  /** Verify cross-field dedup: both fields return the correct vector values. */
  public void testCrossFieldDedupCorrectness() throws IOException {
    float[] vec = {1f, 2f, 3f, 4f};

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, iwc())) {
      for (int i = 0; i < 10; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("field_a", vec, VectorSimilarityFunction.EUCLIDEAN));
        doc.add(new KnnFloatVectorField("field_b", vec, VectorSimilarityFunction.EUCLIDEAN));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader leaf = reader.leaves().get(0).reader();

        // Both fields should return the same vector for every doc
        for (String field : new String[] {"field_a", "field_b"}) {
          FloatVectorValues values = leaf.getFloatVectorValues(field);
          assertEquals(10, values.size());
          KnnVectorValues.DocIndexIterator iter = values.iterator();
          while (iter.nextDoc() != NO_MORE_DOCS) {
            assertArrayEquals(vec, values.vectorValue(iter.index()), 0f);
          }
        }
      }
    }
  }

  /** Verify dedup survives merge of multiple segments. */
  public void testDedupAcrossMerge() throws IOException {
    float[] vec = {1f, 2f, 3f};

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, iwc().setMaxBufferedDocs(5))) {
      // Write 20 docs with the same vector, flushing every 5 → 4 segments
      for (int i = 0; i < 20; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("f", vec, VectorSimilarityFunction.EUCLIDEAN));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        assertEquals(1, reader.leaves().size());
        LeafReader leaf = reader.leaves().get(0).reader();
        FloatVectorValues values = leaf.getFloatVectorValues("f");
        assertEquals(20, values.size());

        KnnVectorValues.DocIndexIterator iter = values.iterator();
        int count = 0;
        while (iter.nextDoc() != NO_MORE_DOCS) {
          assertArrayEquals(vec, values.vectorValue(iter.index()), 0f);
          count++;
        }
        assertEquals(20, count);
      }
    }
  }

  /** Verify that distinct vectors are all preserved (no false dedup). */
  public void testDistinctVectorsPreserved() throws IOException {
    int numDocs = 50;
    int dim = 4;

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, iwc())) {
      float[][] vecs = new float[numDocs][dim];
      for (int i = 0; i < numDocs; i++) {
        Arrays.fill(vecs[i], (float) i);
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("f", vecs[i], VectorSimilarityFunction.EUCLIDEAN));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        FloatVectorValues values = leaf.getFloatVectorValues("f");
        assertEquals(numDocs, values.size());

        KnnVectorValues.DocIndexIterator iter = values.iterator();
        int ord = 0;
        while (iter.nextDoc() != NO_MORE_DOCS) {
          assertArrayEquals(vecs[ord], values.vectorValue(iter.index()), 0f);
          ord++;
        }
        assertEquals(numDocs, ord);
      }
    }
  }

  /** Verify random access to deduplicated vectors works correctly. */
  public void testRandomAccessDedup() throws IOException {
    float[] vecA = {1f, 0f, 0f};
    float[] vecB = {0f, 1f, 0f};

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, iwc())) {
      // Interleave: A, B, A, B, A
      for (int i = 0; i < 5; i++) {
        Document doc = new Document();
        doc.add(
            new KnnFloatVectorField(
                "f", i % 2 == 0 ? vecA : vecB, VectorSimilarityFunction.EUCLIDEAN));
        w.addDocument(doc);
      }
      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        FloatVectorValues values = leaf.getFloatVectorValues("f");

        // Random access: read in non-sequential order
        assertArrayEquals(vecA, values.vectorValue(0), 0f);
        assertArrayEquals(vecA, values.vectorValue(4), 0f);
        assertArrayEquals(vecB, values.vectorValue(1), 0f);
        assertArrayEquals(vecA, values.vectorValue(2), 0f);
        assertArrayEquals(vecB, values.vectorValue(3), 0f);
      }
    }
  }

  /**
   * Verify that vectors with the same Arrays.hashCode (known collision) are NOT falsely deduped.
   * [0,0,0,0] and [2,2,2,2] collide on Arrays.hashCode but must be stored as distinct vectors.
   */
  public void testHashCollisionCorrectness() throws IOException {
    // These two vectors have the same Arrays.hashCode value
    float[] vecA = {0f, 0f, 0f, 0f};
    float[] vecB = {2f, 2f, 2f, 2f};
    assertEquals(
        "precondition: these vectors must collide on Arrays.hashCode",
        Arrays.hashCode(vecA),
        Arrays.hashCode(vecB));

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, iwc())) {
      Document doc1 = new Document();
      doc1.add(new KnnFloatVectorField("f", vecA, VectorSimilarityFunction.EUCLIDEAN));
      w.addDocument(doc1);

      Document doc2 = new Document();
      doc2.add(new KnnFloatVectorField("f", vecB, VectorSimilarityFunction.EUCLIDEAN));
      w.addDocument(doc2);

      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        FloatVectorValues values = leaf.getFloatVectorValues("f");
        assertEquals(2, values.size());

        // Both vectors must be preserved with their correct values
        assertArrayEquals(vecA, values.vectorValue(0), 0f);
        assertArrayEquals(vecB, values.vectorValue(1), 0f);
      }
    }
  }

  /**
   * Simulates the reported scenario: many docs with vector_1, 50% also have vector_2 with the same
   * vector. Cross-field dedup should prevent ~50% raw vector size increase.
   */
  public void testCrossFieldDedupPartialOverlap() throws IOException {
    int numDocs = 1000; // scaled down from 100K for test speed
    int dim = 128;
    Random rng = new Random(42);

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, iwc())) {
      float[][] vecs = new float[numDocs][];
      for (int i = 0; i < numDocs; i++) {
        vecs[i] = new float[dim];
        for (int d = 0; d < dim; d++) {
          vecs[i][d] = rng.nextFloat();
        }
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("vector_1", vecs[i], VectorSimilarityFunction.EUCLIDEAN));
        // 50% of docs also get vector_2 with the SAME vector
        if (i % 2 == 0) {
          doc.add(new KnnFloatVectorField("vector_2", vecs[i], VectorSimilarityFunction.EUCLIDEAN));
        }
        w.addDocument(doc);
      }
      w.forceMerge(1);
      w.commit();

      // Measure .dvd file size
      long dvdSize = 0;
      for (String file : dir.listAll()) {
        if (file.endsWith(DedupFlatVectorsFormat.DATA_EXTENSION)) {
          dvdSize += dir.fileLength(file);
        }
      }

      // Without dedup: 1500 vectors × 128 dims × 4 bytes = 768,000 bytes
      // With dedup: 1000 unique vectors × 128 dims × 4 bytes = 512,000 bytes (plus overhead)
      // The 500 shared vectors should NOT be stored twice.
      long noDedupSize = 1500L * dim * Float.BYTES;
      long expectedDedupSize = 1000L * dim * Float.BYTES;
      assertTrue(
          "dvd file ("
              + dvdSize
              + ") should be close to "
              + expectedDedupSize
              + " (deduped), not "
              + noDedupSize
              + " (no dedup)",
          dvdSize < expectedDedupSize * 1.1); // allow 10% overhead for metadata

      // Verify correctness: both fields return correct vectors
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader leaf = reader.leaves().get(0).reader();
        FloatVectorValues v1 = leaf.getFloatVectorValues("vector_1");
        FloatVectorValues v2 = leaf.getFloatVectorValues("vector_2");
        assertEquals(numDocs, v1.size());
        assertEquals(numDocs / 2, v2.size());

        KnnVectorValues.DocIndexIterator iter1 = v1.iterator();
        KnnVectorValues.DocIndexIterator iter2 = v2.iterator();
        int v2Doc = iter2.nextDoc();
        for (int doc = iter1.nextDoc(); doc != NO_MORE_DOCS; doc = iter1.nextDoc()) {
          float[] val1 = v1.vectorValue(iter1.index());
          assertNotNull(val1);
          // For docs that have vector_2, verify it matches vector_1
          if (doc == v2Doc) {
            float[] val2 = v2.vectorValue(iter2.index());
            assertArrayEquals("vector_2 should match vector_1 for doc " + doc, val1, val2, 0f);
            v2Doc = iter2.nextDoc();
          }
        }
      }
    }
  }
}
