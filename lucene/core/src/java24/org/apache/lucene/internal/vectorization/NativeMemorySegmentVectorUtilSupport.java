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

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

@SuppressWarnings("restricted")
class NativeMemorySegmentVectorUtilSupport implements MemorySegmentVectorUtilSupport {
  private static final PanamaVectorUtilSupport PANAMA = new PanamaVectorUtilSupport();

  static {
    System.loadLibrary("native_vector_util");
  }

  private static final MethodHandle DOT_PRODUCT_BYTES =
      Linker.nativeLinker()
          .downcallHandle(
              SymbolLookup.loaderLookup().findOrThrow("dot_product_bytes"),
              FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT),
              Linker.Option.critical(true));

  @Override
  public int dotProduct(MemorySegment a, MemorySegment b, int limit) {
    assert a.byteSize() >= limit;
    assert b.byteSize() >= limit;
    try {
      return (int) DOT_PRODUCT_BYTES.invokeExact(a, b, limit);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private static final MethodHandle COSINE_BYTES =
      Linker.nativeLinker()
          .downcallHandle(
              SymbolLookup.loaderLookup().findOrThrow("cosine_bytes"),
              FunctionDescriptor.of(JAVA_FLOAT, ADDRESS, ADDRESS, JAVA_INT),
              Linker.Option.critical(true));

  @Override
  public float cosine(MemorySegment a, MemorySegment b, int limit) {
    assert a.byteSize() >= limit;
    assert b.byteSize() >= limit;
    try {
      return (float) COSINE_BYTES.invokeExact(a, b, limit);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private static final MethodHandle SQUARE_DISTANCE_BYTES =
      Linker.nativeLinker()
          .downcallHandle(
              SymbolLookup.loaderLookup().findOrThrow("square_distance_bytes"),
              FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT),
              Linker.Option.critical(true));

  @Override
  public int squareDistance(MemorySegment a, MemorySegment b, int limit) {
    assert a.byteSize() >= limit;
    assert b.byteSize() >= limit;
    try {
      return (int) SQUARE_DISTANCE_BYTES.invokeExact(a, b, limit);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  @Override
  public float dotProduct(float[] a, float[] b) {
    // TODO: Native equivalent, delegate to Panama for now!
    return PANAMA.dotProduct(a, b);
  }

  @Override
  public float cosine(float[] v1, float[] v2) {
    // TODO: Native equivalent, delegate to Panama for now!
    return PANAMA.cosine(v1, v2);
  }

  @Override
  public float squareDistance(float[] a, float[] b) {
    // TODO: Native equivalent, delegate to Panama for now!
    return PANAMA.squareDistance(a, b);
  }

  private static final MethodHandle INT4_DOT_PRODUCT_BYTES =
      Linker.nativeLinker()
          .downcallHandle(
              SymbolLookup.loaderLookup().findOrThrow("int4_dot_product_bytes"),
              FunctionDescriptor.of(
                  JAVA_INT, ADDRESS, JAVA_BOOLEAN, ADDRESS, JAVA_BOOLEAN, JAVA_INT),
              Linker.Option.critical(true));

  @Override
  public int int4DotProduct(byte[] a, boolean apacked, byte[] b, boolean bpacked) {
    try {
      return (int)
          INT4_DOT_PRODUCT_BYTES.invokeExact(
              MemorySegment.ofArray(a), apacked, MemorySegment.ofArray(b), bpacked, a.length);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  @Override
  public int findNextGEQ(int[] buffer, int target, int from, int to) {
    // TODO: Native equivalent, delegate to Panama for now!
    return PANAMA.findNextGEQ(buffer, target, from, to);
  }

  @Override
  public long int4BitDotProduct(byte[] int4Quantized, byte[] binaryQuantized) {
    // TODO: Native equivalent, delegate to Panama for now!
    return PANAMA.int4BitDotProduct(int4Quantized, binaryQuantized);
  }

  @Override
  public float minMaxScalarQuantize(
      float[] vector, byte[] dest, float scale, float alpha, float minQuantile, float maxQuantile) {
    // TODO: Native equivalent, delegate to Panama for now!
    return PANAMA.minMaxScalarQuantize(vector, dest, scale, alpha, minQuantile, maxQuantile);
  }

  @Override
  public float recalculateScalarQuantizationOffset(
      byte[] vector,
      float oldAlpha,
      float oldMinQuantile,
      float scale,
      float alpha,
      float minQuantile,
      float maxQuantile) {
    // TODO: Native equivalent, delegate to Panama for now!
    return PANAMA.recalculateScalarQuantizationOffset(
        vector, oldAlpha, oldMinQuantile, scale, alpha, minQuantile, maxQuantile);
  }

  @Override
  public int filterByScore(
      int[] docBuffer, double[] scoreBuffer, double minScoreInclusive, int upTo) {
    // TODO: Native equivalent, delegate to Panama for now!
    return PANAMA.filterByScore(docBuffer, scoreBuffer, minScoreInclusive, upTo);
  }

  @Override
  public float[] l2normalize(float[] v, boolean throwOnZero) {
    // TODO: Native equivalent, delegate to Panama for now!
    return PANAMA.l2normalize(v, throwOnZero);
  }
}
