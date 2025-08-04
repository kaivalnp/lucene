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

package org.apache.lucene.internal.vectorization;

import java.lang.foreign.MemorySegment;
import java.util.logging.Logger;

interface MemorySegmentVectorUtilSupport extends VectorUtilSupport {
  MemorySegmentVectorUtilSupport INSTANCE = lookup();

  private static MemorySegmentVectorUtilSupport lookup() {
    Logger logger = Logger.getLogger(MemorySegmentVectorUtilSupport.class.getName());
    if (Boolean.getBoolean("org.apache.lucene.internal.vectorization.useNative")) {
      try {
        NativeMemorySegmentVectorUtilSupport instance = new NativeMemorySegmentVectorUtilSupport();
        logger.warning("Using native vectorization.");
        return instance;
      } catch (LinkageError e) {
        logger.warning("Native vectorization not available, fallback to Panama.");
      }
    }
    return new PanamaVectorUtilSupport();
  }

  /** Score two off-heap vectors. */
  int dotProduct(MemorySegment a, MemorySegment b, int limit);

  /** Score two off-heap vectors. */
  float cosine(MemorySegment a, MemorySegment b, int limit);

  /** Score two off-heap vectors. */
  int squareDistance(MemorySegment a, MemorySegment b, int limit);

  default int dotProduct(byte[] a, byte[] b) {
    assert a.length == b.length;
    return dotProduct(MemorySegment.ofArray(a), MemorySegment.ofArray(b), a.length);
  }

  /**
   * Score an on-heap vector against an off-heap vector. This may be optimized for some
   * implementations.
   */
  default int dotProduct(byte[] a, MemorySegment b) {
    assert a.length == b.byteSize();
    return dotProduct(MemorySegment.ofArray(a), b, a.length);
  }

  /** Score two off-heap vectors. This may be optimized for some implementations. */
  default int dotProduct(MemorySegment a, MemorySegment b) {
    assert a.byteSize() == b.byteSize();
    return dotProduct(a, b, Math.toIntExact(a.byteSize()));
  }

  default float cosine(byte[] a, byte[] b) {
    assert a.length == b.length;
    return cosine(MemorySegment.ofArray(a), MemorySegment.ofArray(b), a.length);
  }

  /**
   * Score an on-heap vector against an off-heap vector. This may be optimized for some
   * implementations.
   */
  default float cosine(byte[] a, MemorySegment b) {
    assert a.length == b.byteSize();
    return cosine(MemorySegment.ofArray(a), b, a.length);
  }

  /** Score two off-heap vectors. This may be optimized for some implementations. */
  default float cosine(MemorySegment a, MemorySegment b) {
    assert a.byteSize() == b.byteSize();
    return cosine(a, b, Math.toIntExact(a.byteSize()));
  }

  default int squareDistance(byte[] a, byte[] b) {
    assert a.length == b.length;
    return squareDistance(MemorySegment.ofArray(a), MemorySegment.ofArray(b), a.length);
  }

  /**
   * Score an on-heap vector against an off-heap vector. This may be optimized for some
   * implementations.
   */
  default int squareDistance(byte[] a, MemorySegment b) {
    assert a.length == b.byteSize();
    return squareDistance(MemorySegment.ofArray(a), b, a.length);
  }

  /** Score two off-heap vectors. This may be optimized for some implementations. */
  default int squareDistance(MemorySegment a, MemorySegment b) {
    assert a.byteSize() == b.byteSize();
    return squareDistance(a, b, Math.toIntExact(a.byteSize()));
  }
}
