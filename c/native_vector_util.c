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

#include "native_vector_util.h"

int32_t dot_product_bytes(int8_t* vec1, int8_t* vec2, int32_t limit) {
  int32_t result = 0;
  for (int32_t i = 0; i < limit; i++) {
    result += vec1[i] * vec2[i];
  }
  return result;
}

int32_t int4_dot_product_bytes_single_packed(uint8_t* packed, uint8_t* unpacked, int32_t packed_limit) {
  uint32_t result = 0;
  for (int32_t i = 0; i < packed_limit; i++) {
    result += (packed[i] >> 4) * unpacked[i];
  }
  for (int32_t i = 0; i < packed_limit; i++) {
    result += (packed[i] & 0xF) * unpacked[i + packed_limit];
  }
  return result;
}

int32_t int4_dot_product_bytes(uint8_t* vec1, bool vec1_packed, uint8_t* vec2, bool vec2_packed, int32_t limit) {
  if (vec1_packed) {
    return int4_dot_product_bytes_single_packed(vec1, vec2, limit);
  } else if (vec2_packed) {
    return int4_dot_product_bytes_single_packed(vec2, vec1, limit >> 1);
  } else {
    return dot_product_bytes(vec1, vec2, limit);
  }
}

float32_t cosine_bytes(int8_t* vec1, int8_t* vec2, int32_t limit) {
  int32_t sum = 0, norm1 = 0, norm2 = 0;
  for (int32_t i = 0; i < limit; i++) {
    sum += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  return sum / (float32_t) (norm1 * norm2);
}

int32_t square_distance_bytes(int8_t* vec1, int8_t* vec2, int32_t limit) {
  uint32_t result = 0;
  for (int32_t i = 0; i < limit; i++) {
    int32_t diff = vec1[i] - vec2[i];
    result += diff * diff;
  }
  return result;
}
