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

package org.apache.lucene.search;

import java.io.IOException;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.hnsw.NeighborQueue;

/** Placeholder javadocs. */
public record BoundedCandidatesKnnCollectorManager(KnnCollectorManager delegate, int efSearch)
    implements KnnCollectorManager {
  @Override
  public KnnCollector newCollector(int visitedLimit, LeafReaderContext context) throws IOException {
    return new KnnCollectorImpl(delegate.newCollector(visitedLimit, context), efSearch);
  }

  private record KnnCollectorImpl(KnnCollector delegate, int efSearch) implements KnnCollector {
    @Override
    public boolean earlyTerminated() {
      return delegate.earlyTerminated();
    }

    @Override
    public void incVisitedCount(int count) {
      delegate.incVisitedCount(count);
    }

    @Override
    public long visitedCount() {
      return delegate.visitedCount();
    }

    @Override
    public long visitLimit() {
      return delegate.visitLimit();
    }

    @Override
    public int k() {
      return delegate.k();
    }

    @Override
    public boolean collect(int docId, float similarity) {
      return delegate.collect(docId, similarity);
    }

    @Override
    public float minCompetitiveSimilarity() {
      return delegate.minCompetitiveSimilarity();
    }

    @Override
    public TopDocs topDocs() {
      return delegate.topDocs();
    }

    @Override
    public NeighborQueue candidates() {
      return new NeighborQueue(efSearch, true) {
        @Override
        public void add(int newNode, float newScore) {
          insertWithOverflow(newNode, newScore);
        }
      };
    }
  }
}
