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

#include <stdint.h>
#include <stdbool.h>

#define float32_t float

int32_t dot_product_bytes(int8_t* vec1, int8_t* vec2, int32_t limit);

int32_t int4_dot_product_bytes(uint8_t* vec1, bool vec1_packed, uint8_t* vec2, bool vec2_packed, int32_t limit);

float32_t cosine_bytes(int8_t* vec1, int8_t* vec2, int32_t limit);

int32_t square_distance_bytes(int8_t* vec1, int8_t* vec2, int32_t limit);
