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

package org.apache.lucene.luke.util;

import java.util.logging.Level;
import java.util.logging.Logger;

/** Logger factory. This configures log interceptors for the GUI. */
public class LoggerFactory {
  @SuppressWarnings("NonFinalStaticField")
  public static CircularLogBufferHandler circularBuffer;

  public static void initGuiLogging() {
    if (circularBuffer != null) {
      throw new RuntimeException("Double-initialization?");
    }

    circularBuffer = new CircularLogBufferHandler();
    circularBuffer.setLevel(Level.FINEST);

    // Only capture events from Lucene logger hierarchy.
    var luceneRoot = Logger.getLogger("org.apache.lucene");
    luceneRoot.setLevel(Level.FINEST);
    luceneRoot.addHandler(circularBuffer);
  }

  public static Logger getLogger(Class<?> clazz) {
    return Logger.getLogger(clazz.getName());
  }
}
