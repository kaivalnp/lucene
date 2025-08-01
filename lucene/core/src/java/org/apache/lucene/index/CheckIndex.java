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
package org.apache.lucene.index;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.NumberFormat;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.TermVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DocumentStoredFieldVisitor;
import org.apache.lucene.index.CheckIndex.Status.DocValuesStatus;
import org.apache.lucene.index.PointValues.IntersectVisitor;
import org.apache.lucene.index.PointValues.Relation;
import org.apache.lucene.internal.hppc.IntIntHashMap;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.LeafFieldComparator;
import org.apache.lucene.search.Pruning;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.Lock;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.ArrayUtil.ByteArrayComparator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefBuilder;
import org.apache.lucene.util.CommandLineUtil;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOBooleanSupplier;
import org.apache.lucene.util.IOFunction;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.LongBitSet;
import org.apache.lucene.util.NamedThreadFactory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.automaton.Automata;
import org.apache.lucene.util.automaton.Automaton;
import org.apache.lucene.util.automaton.ByteRunAutomaton;
import org.apache.lucene.util.automaton.CompiledAutomaton;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.hnsw.HnswGraph;

/**
 * Basic tool and API to check the health of an index and write a new segments file that removes
 * reference to problematic segments.
 *
 * <p>As this tool checks every byte in the index, on a large index it can take quite a long time to
 * run.
 *
 * @lucene.experimental Please make a complete backup of your index before using this to exorcise
 *     corrupted documents from your index!
 */
public final class CheckIndex implements Closeable {

  private final Directory dir;
  private final Lock writeLock;
  private final NumberFormat nf = NumberFormat.getInstance(Locale.ROOT);
  private PrintStream infoStream;
  private volatile boolean closed;

  /**
   * Returned from {@link #checkIndex()} detailing the health and status of the index.
   *
   * @lucene.experimental
   */
  public static class Status {

    Status() {}

    /** True if no problems were found with the index. */
    public boolean clean;

    /** True if we were unable to locate and load the segments_N file. */
    public boolean missingSegments;

    /** Name of latest segments_N file in the index. */
    public String segmentsFileName;

    /** Number of segments in the index. */
    public int numSegments;

    /**
     * Empty unless you passed specific segments list to check as optional 3rd argument.
     *
     * @see CheckIndex#checkIndex(List)
     */
    public List<String> segmentsChecked = new ArrayList<>();

    /** True if the index was created with a newer version of Lucene than the CheckIndex tool. */
    public boolean toolOutOfDate;

    /** List of {@link SegmentInfoStatus} instances, detailing status of each segment. */
    public List<SegmentInfoStatus> segmentInfos = new ArrayList<>();

    /** Directory index is in. */
    public Directory dir;

    /**
     * SegmentInfos instance containing only segments that had no problems (this is used with the
     * {@link CheckIndex#exorciseIndex} method to repair the index).
     */
    SegmentInfos newSegments;

    /** How many documents will be lost to bad segments. */
    public int totLoseDocCount;

    /** How many bad segments were found. */
    public int numBadSegments;

    /**
     * True if we checked only specific segments ({@link #checkIndex(List)} was called with non-null
     * argument).
     */
    public boolean partial;

    /** The greatest segment name. */
    public long maxSegmentName;

    /** Whether the SegmentInfos.counter is greater than any of the segments' names. */
    public boolean validCounter;

    /** Holds the userData of the last commit in the index */
    public Map<String, String> userData;

    /**
     * Holds the status of each segment in the index. See {@link #segmentInfos}.
     *
     * @lucene.experimental
     */
    public static class SegmentInfoStatus {

      SegmentInfoStatus() {}

      /** Name of the segment. */
      public String name;

      /** Codec used to read this segment. */
      public Codec codec;

      /** Document count (does not take deletions into account). */
      public int maxDoc;

      /** True if segment is compound file format. */
      public boolean compound;

      /** Number of files referenced by this segment. */
      public int numFiles;

      /** Net size (MB) of the files referenced by this segment. */
      public double sizeMB;

      /** True if this segment has pending deletions. */
      public boolean hasDeletions;

      /** Current deletions generation. */
      public long deletionsGen;

      /** True if we were able to open a CodecReader on this segment. */
      public boolean openReaderPassed;

      /** doc count in this segment */
      public int toLoseDocCount;

      /**
       * Map that includes certain debugging details that IndexWriter records into each segment it
       * creates
       */
      public Map<String, String> diagnostics;

      /** Status for testing of livedocs */
      public LiveDocStatus liveDocStatus;

      /** Status for testing of field infos */
      public FieldInfoStatus fieldInfoStatus;

      /** Status for testing of field norms (null if field norms could not be tested). */
      public FieldNormStatus fieldNormStatus;

      /** Status for testing of indexed terms (null if indexed terms could not be tested). */
      public TermIndexStatus termIndexStatus;

      /** Status for testing of stored fields (null if stored fields could not be tested). */
      public StoredFieldStatus storedFieldStatus;

      /** Status for testing of term vectors (null if term vectors could not be tested). */
      public TermVectorStatus termVectorStatus;

      /** Status for testing of DocValues (null if DocValues could not be tested). */
      public DocValuesStatus docValuesStatus;

      /** Status for testing of PointValues (null if PointValues could not be tested). */
      public PointsStatus pointsStatus;

      /** Status of index sort */
      public IndexSortStatus indexSortStatus;

      /** Status of vectors */
      public VectorValuesStatus vectorValuesStatus;

      /** Status of HNSW graph */
      public HnswGraphsStatus hnswGraphsStatus;

      /** Status of soft deletes */
      public SoftDeletesStatus softDeletesStatus;

      /** Exception thrown during segment test (null on success) */
      public Throwable error;
    }

    /** Status from testing livedocs */
    public static final class LiveDocStatus {
      private LiveDocStatus() {}

      /** Number of deleted documents. */
      public int numDeleted;

      /** Exception thrown during term index test (null on success) */
      public Throwable error;
    }

    /** Status from testing field infos. */
    public static final class FieldInfoStatus {
      private FieldInfoStatus() {}

      /** Number of fields successfully tested */
      public long totFields = 0L;

      /** Exception thrown during term index test (null on success) */
      public Throwable error;
    }

    /** Status from testing field norms. */
    public static final class FieldNormStatus {
      private FieldNormStatus() {}

      /** Number of fields successfully tested */
      public long totFields = 0L;

      /** Exception thrown during term index test (null on success) */
      public Throwable error;
    }

    /** Status from testing term index. */
    public static final class TermIndexStatus {

      TermIndexStatus() {}

      /** Number of terms with at least one live doc. */
      public long termCount = 0L;

      /** Number of terms with zero live docs. */
      public long delTermCount = 0L;

      /** Total frequency across all terms. */
      public long totFreq = 0L;

      /** Total number of positions. */
      public long totPos = 0L;

      /** Exception thrown during term index test (null on success) */
      public Throwable error;

      /**
       * Holds details of block allocations in the block tree terms dictionary (this is only set if
       * the {@link PostingsFormat} for this segment uses block tree).
       */
      public Map<String, Object> blockTreeStats = null;
    }

    /** Status from testing stored fields. */
    public static final class StoredFieldStatus {

      StoredFieldStatus() {}

      /** Number of documents tested. */
      public int docCount = 0;

      /** Total number of stored fields tested. */
      public long totFields = 0;

      /** Exception thrown during stored fields test (null on success) */
      public Throwable error;
    }

    /** Status from testing stored fields. */
    public static final class TermVectorStatus {

      TermVectorStatus() {}

      /** Number of documents tested. */
      public int docCount = 0;

      /** Total number of term vectors tested. */
      public long totVectors = 0;

      /** Exception thrown during term vector test (null on success) */
      public Throwable error;
    }

    /** Status from testing DocValues */
    public static final class DocValuesStatus {

      DocValuesStatus() {}

      /** Total number of docValues tested. */
      public long totalValueFields;

      /** Total number of numeric fields */
      public long totalNumericFields;

      /** Total number of binary fields */
      public long totalBinaryFields;

      /** Total number of sorted fields */
      public long totalSortedFields;

      /** Total number of sortednumeric fields */
      public long totalSortedNumericFields;

      /** Total number of sortedset fields */
      public long totalSortedSetFields;

      /** Total number of skipping index tested. */
      public long totalSkippingIndex;

      /** Exception thrown during doc values test (null on success) */
      public Throwable error;
    }

    /** Status from testing PointValues */
    public static final class PointsStatus {

      PointsStatus() {}

      /** Total number of values points tested. */
      public long totalValuePoints;

      /** Total number of fields with points. */
      public int totalValueFields;

      /** Exception thrown during point values test (null on success) */
      public Throwable error;
    }

    /** Status from testing vector values */
    public static final class VectorValuesStatus {

      VectorValuesStatus() {}

      /** Total number of vector values tested. */
      public long totalVectorValues;

      /** Total number of fields with vectors. */
      public int totalKnnVectorFields;

      /** Exception thrown during vector values test (null on success) */
      public Throwable error;
    }

    /** Status from testing a single HNSW graph */
    public static final class HnswGraphStatus {

      HnswGraphStatus() {}

      /** Number of nodes at each level */
      public List<Integer> numNodesAtLevel;

      /** Connectedness at each level represented as a fraction */
      public List<String> connectednessAtLevel;
    }

    /** Status from testing all HNSW graphs */
    public static final class HnswGraphsStatus {

      HnswGraphsStatus() {
        this.hnswGraphsStatusByField = new HashMap<>();
      }

      /** Status of the HNSW graph keyed with field name */
      public Map<String, HnswGraphStatus> hnswGraphsStatusByField;

      /** Exception thrown during term index test (null on success) */
      public Throwable error;
    }

    /** Status from testing index sort */
    public static final class IndexSortStatus {
      IndexSortStatus() {}

      /** Exception thrown during term index test (null on success) */
      public Throwable error;
    }

    /** Status from testing soft deletes */
    public static final class SoftDeletesStatus {
      SoftDeletesStatus() {}

      /** Exception thrown during soft deletes test (null on success) */
      public Throwable error;
    }
  }

  /** Create a new CheckIndex on the directory. */
  public CheckIndex(Directory dir) throws IOException {
    this(dir, dir.obtainLock(IndexWriter.WRITE_LOCK_NAME));
  }

  /**
   * Expert: create a directory with the specified lock. This should really not be used except for
   * unit tests!!!! It exists only to support special tests (such as TestIndexWriterExceptions*),
   * that would otherwise be more complicated to debug if they had to close the writer for each
   * check.
   */
  public CheckIndex(Directory dir, Lock writeLock) {
    this.dir = dir;
    this.writeLock = writeLock;
    this.infoStream = null;
  }

  private void ensureOpen() {
    if (closed) {
      throw new AlreadyClosedException("this instance is closed");
    }
  }

  @Override
  public void close() throws IOException {
    closed = true;
    IOUtils.close(writeLock);
  }

  private int level;

  /**
   * Sets Level, the higher the value, the more additional checks are performed. This will likely
   * drastically increase time it takes to run CheckIndex! See {@link Level}
   */
  public void setLevel(int v) {
    Level.checkIfLevelInBounds(v);
    level = v;
  }

  /** See {@link #setLevel}. */
  public int getLevel() {
    return level;
  }

  private boolean failFast;

  /**
   * If true, just throw the original exception immediately when corruption is detected, rather than
   * continuing to iterate to other segments looking for more corruption.
   */
  public void setFailFast(boolean v) {
    failFast = v;
  }

  /** See {@link #setFailFast}. */
  public boolean getFailFast() {
    return failFast;
  }

  private boolean verbose;

  /** Set threadCount used for parallelizing index integrity checking. */
  public void setThreadCount(int tc) {
    if (tc <= 0) {
      throw new IllegalArgumentException(
          "setThreadCount requires a number larger than 0, but got: " + tc);
    }
    threadCount = tc;
  }

  private int threadCount = Runtime.getRuntime().availableProcessors();

  /**
   * Set infoStream where messages should go. If null, no messages are printed. If verbose is true
   * then more details are printed.
   */
  public void setInfoStream(PrintStream out, boolean verbose) {
    infoStream = out;
    this.verbose = verbose;
  }

  /** Set infoStream where messages should go. See {@link #setInfoStream(PrintStream,boolean)}. */
  public void setInfoStream(PrintStream out) {
    setInfoStream(out, false);
  }

  private static void msg(PrintStream out, ByteArrayOutputStream msg) {
    if (out != null) {
      out.println(msg.toString(UTF_8));
    }
  }

  private static void msg(PrintStream out, String msg) {
    if (out != null) {
      out.println(msg);
    }
  }

  /**
   * Returns a {@link Status} instance detailing the state of the index.
   *
   * <p>As this method checks every byte in the index, on a large index it can take quite a long
   * time to run.
   *
   * <p><b>WARNING</b>: make sure you only call this when the index is not opened by any writer.
   */
  public Status checkIndex() throws IOException {
    return checkIndex(null);
  }

  /**
   * Returns a {@link Status} instance detailing the state of the index.
   *
   * @param onlySegments list of specific segment names to check
   *     <p>As this method checks every byte in the specified segments, on a large index it can take
   *     quite a long time to run.
   */
  public Status checkIndex(List<String> onlySegments) throws IOException {
    ExecutorService executorService = null;

    // if threadCount == 1, then no executor is created and use the main thread to do index checking
    // sequentially
    if (threadCount > 1) {
      executorService =
          Executors.newFixedThreadPool(threadCount, new NamedThreadFactory("async-check-index"));
    }

    msg(infoStream, "Checking index with threadCount: " + threadCount);
    try {
      return checkIndex(onlySegments, executorService);
    } finally {
      if (executorService != null) {
        executorService.shutdown();
        try {
          executorService.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
          msg(
              infoStream,
              "ERROR: Interrupted exception occurred when shutting down executor service");
          if (infoStream != null) e.printStackTrace(infoStream);
        } finally {
          executorService.shutdownNow();
        }
      }
    }
  }

  /**
   * Returns a {@link Status} instance detailing the state of the index.
   *
   * <p>This method allows caller to pass in customized ExecutorService to speed up the check.
   *
   * <p><b>WARNING</b>: make sure you only call this when the index is not opened by any writer.
   */
  public Status checkIndex(List<String> onlySegments, ExecutorService executorService)
      throws IOException {
    ensureOpen();
    long startNS = System.nanoTime();

    Status result = new Status();
    result.dir = dir;
    String[] files = dir.listAll();
    String lastSegmentsFile = SegmentInfos.getLastCommitSegmentsFileName(files);
    if (lastSegmentsFile == null) {
      throw new IndexNotFoundException(
          "no segments* file found in " + dir + ": files: " + Arrays.toString(files));
    }

    // https://github.com/apache/lucene/issues/7820: also attempt to open any older commit
    // points (segments_N), which will catch certain corruption like missing _N.si files
    // for segments not also referenced by the newest commit point (which was already
    // loaded, successfully, above).  Note that we do not do a deeper check of segments
    // referenced ONLY by these older commit points, because such corruption would not
    // prevent a new IndexWriter from opening on the newest commit point.  but it is still
    // corruption, e.g. a reader opened on those old commit points can hit corruption
    // exceptions which we (still) will not detect here.  progress not perfection!

    SegmentInfos lastCommit = null;

    List<String> allSegmentsFiles = new ArrayList<>();
    for (String fileName : files) {
      if (fileName.startsWith(IndexFileNames.SEGMENTS)
          && fileName.equals(SegmentInfos.OLD_SEGMENTS_GEN) == false) {
        allSegmentsFiles.add(fileName);
      }
    }

    // Sort descending by generation so that we always attempt to read the last commit first.  This
    // way if an index has a broken last commit AND a broken old commit, we report the last commit
    // error first:
    allSegmentsFiles.sort(
        (a, b) -> {
          long genA = SegmentInfos.generationFromSegmentsFileName(a);
          long genB = SegmentInfos.generationFromSegmentsFileName(b);

          // reversed natural sort (largest generation first):
          return -Long.compare(genA, genB);
        });

    for (String fileName : allSegmentsFiles) {

      boolean isLastCommit = fileName.equals(lastSegmentsFile);

      SegmentInfos infos;

      try {
        // Do not use SegmentInfos.read(Directory) since the spooky
        // retrying it does is not necessary here (we hold the write lock):
        // always open old indices if codecs are around
        infos = SegmentInfos.readCommit(dir, fileName, 0);
      } catch (Throwable t) {
        if (failFast) {
          throw IOUtils.rethrowAlways(t);
        }

        String message;

        if (isLastCommit) {
          message =
              "ERROR: could not read latest commit point from segments file \""
                  + fileName
                  + "\" in directory";
        } else {
          message =
              "ERROR: could not read old (not latest) commit point segments file \""
                  + fileName
                  + "\" in directory";
        }
        msg(infoStream, message);
        result.missingSegments = true;
        if (infoStream != null) {
          t.printStackTrace(infoStream);
        }
        return result;
      }

      if (isLastCommit) {
        // record the latest commit point: we will deeply check all segments referenced by it
        lastCommit = infos;
      }
    }

    // we know there is a lastSegmentsFileName, so we must've attempted to load it in the above for
    // loop.  if it failed to load, we threw the exception (fastFail == true) or we returned the
    // failure (fastFail == false).  so if we get here, we should // always have a valid lastCommit:
    assert lastCommit != null;

    if (lastCommit == null) {
      msg(infoStream, "ERROR: could not read any segments file in directory");
      result.missingSegments = true;
      return result;
    }

    if (infoStream != null) {
      int maxDoc = 0;
      int delCount = 0;
      for (SegmentCommitInfo info : lastCommit) {
        maxDoc += info.info.maxDoc();
        delCount += info.getDelCount();
      }
      infoStream.printf(
          Locale.ROOT,
          "%.2f%% total deletions; %d documents; %d deletions%n",
          100. * delCount / maxDoc,
          maxDoc,
          delCount);
    }

    // find the oldest and newest segment versions
    Version oldest = null;
    Version newest = null;
    String oldSegs = null;
    for (SegmentCommitInfo si : lastCommit) {
      Version version = si.info.getVersion();
      if (version == null) {
        // pre-3.1 segment
        oldSegs = "pre-3.1";
      } else {
        if (oldest == null || version.onOrAfter(oldest) == false) {
          oldest = version;
        }
        if (newest == null || version.onOrAfter(newest)) {
          newest = version;
        }
      }
    }

    final int numSegments = lastCommit.size();
    final String segmentsFileName = lastCommit.getSegmentsFileName();
    result.segmentsFileName = segmentsFileName;
    result.numSegments = numSegments;
    result.userData = lastCommit.getUserData();
    String userDataString;
    if (lastCommit.getUserData().size() > 0) {
      userDataString = " userData=" + lastCommit.getUserData();
    } else {
      userDataString = "";
    }

    String versionString = "";
    if (oldSegs != null) {
      if (newest != null) {
        versionString = "versions=[" + oldSegs + " .. " + newest + "]";
      } else {
        versionString = "version=" + oldSegs;
      }
    } else if (newest != null) { // implies oldest != null
      versionString =
          oldest.equals(newest)
              ? ("version=" + oldest)
              : ("versions=[" + oldest + " .. " + newest + "]");
    }

    msg(
        infoStream,
        "Segments file="
            + segmentsFileName
            + " numSegments="
            + numSegments
            + " "
            + versionString
            + " id="
            + StringHelper.idToString(lastCommit.getId())
            + userDataString);

    if (onlySegments != null) {
      result.partial = true;
      if (infoStream != null) {
        infoStream.print("\nChecking only these segments:");
        for (String s : onlySegments) {
          infoStream.print(" " + s);
        }
      }
      result.segmentsChecked.addAll(onlySegments);
      msg(infoStream, ":");
    }

    result.newSegments = lastCommit.clone();
    result.newSegments.clear();
    result.maxSegmentName = -1;

    // checks segments sequentially
    if (executorService == null) {
      for (int i = 0; i < numSegments; i++) {
        final SegmentCommitInfo info = lastCommit.info(i);
        updateMaxSegmentName(result, info);
        if (onlySegments != null && !onlySegments.contains(info.info.name)) {
          continue;
        }

        msg(
            infoStream,
            (1 + i)
                + " of "
                + numSegments
                + ": name="
                + info.info.name
                + " maxDoc="
                + info.info.maxDoc());
        Status.SegmentInfoStatus segmentInfoStatus = testSegment(lastCommit, info, infoStream);

        processSegmentInfoStatusResult(result, info, segmentInfoStatus);
      }
    } else {
      ByteArrayOutputStream[] outputs = new ByteArrayOutputStream[numSegments];
      @SuppressWarnings({"unchecked", "rawtypes"})
      CompletableFuture<Status.SegmentInfoStatus>[] futures = new CompletableFuture[numSegments];

      // checks segments concurrently
      List<SegmentCommitInfo> segmentCommitInfos = new ArrayList<>();
      for (SegmentCommitInfo sci : lastCommit) {
        segmentCommitInfos.add(sci);
      }

      // sort segmentCommitInfos by segment size, as smaller segment tends to finish faster, and
      // hence its output can be printed out faster
      segmentCommitInfos.sort(
          (info1, info2) -> {
            try {
              return Long.compare(info1.sizeInBytes(), info2.sizeInBytes());
            } catch (IOException e) {
              msg(
                  infoStream,
                  "ERROR: IOException occurred when comparing SegmentCommitInfo file sizes");
              if (infoStream != null) e.printStackTrace(infoStream);
              return 0;
            }
          });

      // start larger segments earlier
      for (int i = numSegments - 1; i >= 0; i--) {
        final SegmentCommitInfo info = segmentCommitInfos.get(i);
        updateMaxSegmentName(result, info);
        if (onlySegments != null && !onlySegments.contains(info.info.name)) {
          continue;
        }

        SegmentInfos finalSis = lastCommit;

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        PrintStream stream = new PrintStream(output, true, UTF_8);
        msg(
            stream,
            (1 + i)
                + " of "
                + numSegments
                + ": name="
                + info.info.name
                + " maxDoc="
                + info.info.maxDoc());

        outputs[i] = output;
        futures[i] =
            runAsyncSegmentCheck(() -> testSegment(finalSis, info, stream), executorService);
      }

      for (int i = 0; i < numSegments; i++) {
        SegmentCommitInfo info = segmentCommitInfos.get(i);
        if (onlySegments != null && !onlySegments.contains(info.info.name)) {
          continue;
        }

        ByteArrayOutputStream output = outputs[i];

        // print segment results in order
        Status.SegmentInfoStatus segmentInfoStatus = null;
        try {
          segmentInfoStatus = futures[i].get();
        } catch (InterruptedException e) {
          // the segment test output should come before interrupted exception message that follows,
          // hence it's not emitted from finally clause
          msg(infoStream, output);
          msg(
              infoStream,
              "ERROR: Interrupted exception occurred when getting segment check result for segment "
                  + info.info.name);
          if (infoStream != null) e.printStackTrace(infoStream);
        } catch (ExecutionException e) {
          msg(infoStream, output.toString(UTF_8));

          assert failFast;
          throw new CheckIndexException(
              "Segment " + info.info.name + " check failed.", e.getCause());
        }

        msg(infoStream, output);

        processSegmentInfoStatusResult(result, info, segmentInfoStatus);
      }
    }

    if (0 == result.numBadSegments) {
      result.clean = true;
    } else {
      msg(
          infoStream,
          "WARNING: "
              + result.numBadSegments
              + " broken segments (containing "
              + result.totLoseDocCount
              + " documents) detected");
    }

    result.validCounter = result.maxSegmentName < lastCommit.counter;
    if (result.validCounter == false) {
      result.clean = false;
      result.newSegments.counter = result.maxSegmentName + 1;
      msg(
          infoStream,
          "ERROR: Next segment name counter "
              + lastCommit.counter
              + " is not greater than max segment name "
              + result.maxSegmentName);
    }

    if (result.clean) {
      msg(infoStream, "No problems were detected with this index.\n");
    }

    msg(
        infoStream,
        String.format(Locale.ROOT, "Took %.3f sec total.", nsToSec(System.nanoTime() - startNS)));

    return result;
  }

  private void updateMaxSegmentName(Status result, SegmentCommitInfo info) {
    long segmentName = Long.parseLong(info.info.name.substring(1), Character.MAX_RADIX);
    if (segmentName > result.maxSegmentName) {
      result.maxSegmentName = segmentName;
    }
  }

  private void processSegmentInfoStatusResult(
      Status result, SegmentCommitInfo info, Status.SegmentInfoStatus segmentInfoStatus) {
    result.segmentInfos.add(segmentInfoStatus);
    if (segmentInfoStatus.error != null) {
      result.totLoseDocCount += segmentInfoStatus.toLoseDocCount;
      result.numBadSegments++;
    } else {
      // Keeper
      result.newSegments.add(info.clone());
    }
  }

  private <R> CompletableFuture<R> runAsyncSegmentCheck(
      Callable<R> asyncCallable, ExecutorService executorService) {
    return CompletableFuture.supplyAsync(callableToSupplier(asyncCallable), executorService);
  }

  private <T> Supplier<T> callableToSupplier(Callable<T> callable) {
    return () -> {
      try {
        return callable.call();
      } catch (RuntimeException | Error e) {
        throw e;
      } catch (Throwable e) {
        throw new CompletionException(e);
      }
    };
  }

  private Status.SegmentInfoStatus testSegment(
      SegmentInfos sis, SegmentCommitInfo info, PrintStream infoStream) throws IOException {
    Status.SegmentInfoStatus segInfoStat = new Status.SegmentInfoStatus();
    segInfoStat.name = info.info.name;
    segInfoStat.maxDoc = info.info.maxDoc();

    final Version version = info.info.getVersion();
    if (info.info.maxDoc() <= 0) {
      throw new CheckIndexException(" illegal number of documents: maxDoc=" + info.info.maxDoc());
    }

    int toLoseDocCount = info.info.maxDoc();

    SegmentReader reader = null;

    try {
      msg(infoStream, "    version=" + (version == null ? "3.0" : version));
      msg(infoStream, "    id=" + StringHelper.idToString(info.info.getId()));
      final Codec codec = info.info.getCodec();
      msg(infoStream, "    codec=" + codec);
      segInfoStat.codec = codec;
      msg(infoStream, "    compound=" + info.info.getUseCompoundFile());
      segInfoStat.compound = info.info.getUseCompoundFile();
      msg(infoStream, "    numFiles=" + info.files().size());
      Sort indexSort = info.info.getIndexSort();
      if (indexSort != null) {
        msg(infoStream, "    sort=" + indexSort);
      }
      segInfoStat.numFiles = info.files().size();
      segInfoStat.sizeMB = info.sizeInBytes() / (1024. * 1024.);
      // nf#format is not thread-safe, and would generate random non-valid results in concurrent
      // setting
      synchronized (nf) {
        msg(infoStream, "    size (MB)=" + nf.format(segInfoStat.sizeMB));
      }
      Map<String, String> diagnostics = info.info.getDiagnostics();
      segInfoStat.diagnostics = diagnostics;
      if (diagnostics.size() > 0) {
        msg(infoStream, "    diagnostics = " + diagnostics);
      }

      if (info.hasDeletions() == false) {
        msg(infoStream, "    no deletions");
        segInfoStat.hasDeletions = false;
      } else {
        msg(infoStream, "    has deletions [delGen=" + info.getDelGen() + "]");
        segInfoStat.hasDeletions = true;
        segInfoStat.deletionsGen = info.getDelGen();
      }

      long startOpenReaderNS = System.nanoTime();
      if (infoStream != null) infoStream.print("    test: open reader.........");
      reader = new SegmentReader(info, sis.getIndexCreatedVersionMajor(), IOContext.DEFAULT);
      msg(
          infoStream,
          String.format(
              Locale.ROOT, "OK [took %.3f sec]", nsToSec(System.nanoTime() - startOpenReaderNS)));

      segInfoStat.openReaderPassed = true;

      long startIntegrityNS = System.nanoTime();
      if (infoStream != null) infoStream.print("    test: check integrity.....");
      reader.checkIntegrity();
      msg(
          infoStream,
          String.format(
              Locale.ROOT, "OK [took %.3f sec]", nsToSec(System.nanoTime() - startIntegrityNS)));

      if (reader.maxDoc() != info.info.maxDoc()) {
        throw new CheckIndexException(
            "SegmentReader.maxDoc() "
                + reader.maxDoc()
                + " != SegmentInfo.maxDoc "
                + info.info.maxDoc());
      }

      final int numDocs = reader.numDocs();
      toLoseDocCount = numDocs;

      if (reader.hasDeletions()) {
        if (numDocs != info.info.maxDoc() - info.getDelCount()) {
          throw new CheckIndexException(
              "delete count mismatch: info="
                  + (info.info.maxDoc() - info.getDelCount())
                  + " vs reader="
                  + numDocs);
        }
        if ((info.info.maxDoc() - numDocs) > reader.maxDoc()) {
          throw new CheckIndexException(
              "too many deleted docs: maxDoc()="
                  + reader.maxDoc()
                  + " vs del count="
                  + (info.info.maxDoc() - numDocs));
        }
        if (info.info.maxDoc() - numDocs != info.getDelCount()) {
          throw new CheckIndexException(
              "delete count mismatch: info="
                  + info.getDelCount()
                  + " vs reader="
                  + (info.info.maxDoc() - numDocs));
        }
      } else {
        if (info.getDelCount() != 0) {
          throw new CheckIndexException(
              "delete count mismatch: info="
                  + info.getDelCount()
                  + " vs reader="
                  + (info.info.maxDoc() - numDocs));
        }
      }
      if (level >= Level.MIN_LEVEL_FOR_INTEGRITY_CHECKS) {
        // Test Livedocs
        segInfoStat.liveDocStatus = testLiveDocs(reader, infoStream, failFast);

        // Test Fieldinfos
        segInfoStat.fieldInfoStatus = testFieldInfos(reader, infoStream, failFast);

        // Test Field Norms
        segInfoStat.fieldNormStatus = testFieldNorms(reader, infoStream, failFast);

        // Test the Term Index
        segInfoStat.termIndexStatus = testPostings(reader, infoStream, verbose, level, failFast);

        // Test Stored Fields
        segInfoStat.storedFieldStatus = testStoredFields(reader, infoStream, failFast);

        // Test Term Vectors
        segInfoStat.termVectorStatus =
            testTermVectors(reader, infoStream, verbose, level, failFast);

        // Test Docvalues
        segInfoStat.docValuesStatus = testDocValues(reader, infoStream, failFast);

        // Test PointValues
        segInfoStat.pointsStatus = testPoints(reader, infoStream, failFast);

        // Test FloatVectorValues and ByteVectorValues
        segInfoStat.vectorValuesStatus = testVectors(reader, infoStream, failFast);

        // Test HNSW graph
        segInfoStat.hnswGraphsStatus = testHnswGraphs(reader, infoStream, failFast);

        // Test Index Sort
        if (indexSort != null) {
          segInfoStat.indexSortStatus = testSort(reader, indexSort, infoStream, failFast);
        }

        // Test Soft Deletes
        final String softDeletesField = reader.getFieldInfos().getSoftDeletesField();
        if (softDeletesField != null) {
          segInfoStat.softDeletesStatus =
              checkSoftDeletes(softDeletesField, info, reader, infoStream, failFast);
        }

        // Rethrow the first exception we encountered
        //  This will cause stats for failed segments to be incremented properly
        // We won't be able to (easily) stop check running in another thread, so we may as well
        // wait for all of them to complete before we proceed, and that we don't throw
        // CheckIndexException
        // below while the segment part check may still print out messages
        if (segInfoStat.liveDocStatus.error != null) {
          throw new CheckIndexException("Live docs test failed", segInfoStat.liveDocStatus.error);
        } else if (segInfoStat.fieldInfoStatus.error != null) {
          throw new CheckIndexException(
              "Field Info test failed", segInfoStat.fieldInfoStatus.error);
        } else if (segInfoStat.fieldNormStatus.error != null) {
          throw new CheckIndexException(
              "Field Norm test failed", segInfoStat.fieldNormStatus.error);
        } else if (segInfoStat.termIndexStatus.error != null) {
          throw new CheckIndexException(
              "Term Index test failed", segInfoStat.termIndexStatus.error);
        } else if (segInfoStat.storedFieldStatus.error != null) {
          throw new CheckIndexException(
              "Stored Field test failed", segInfoStat.storedFieldStatus.error);
        } else if (segInfoStat.termVectorStatus.error != null) {
          throw new CheckIndexException(
              "Term Vector test failed", segInfoStat.termVectorStatus.error);
        } else if (segInfoStat.docValuesStatus.error != null) {
          throw new CheckIndexException("DocValues test failed", segInfoStat.docValuesStatus.error);
        } else if (segInfoStat.pointsStatus.error != null) {
          throw new CheckIndexException("Points test failed", segInfoStat.pointsStatus.error);
        } else if (segInfoStat.vectorValuesStatus.error != null) {
          throw new CheckIndexException(
              "Vectors test failed", segInfoStat.vectorValuesStatus.error);
        } else if (segInfoStat.indexSortStatus != null
            && segInfoStat.indexSortStatus.error != null) {
          throw new CheckIndexException(
              "Index Sort test failed", segInfoStat.indexSortStatus.error);
        } else if (segInfoStat.softDeletesStatus != null
            && segInfoStat.softDeletesStatus.error != null) {
          throw new CheckIndexException(
              "Soft Deletes test failed", segInfoStat.softDeletesStatus.error);
        }
      }

      msg(infoStream, "");
    } catch (Throwable t) {
      if (failFast) {
        throw IOUtils.rethrowAlways(t);
      }
      segInfoStat.error = t;
      segInfoStat.toLoseDocCount = toLoseDocCount;
      msg(infoStream, "FAILED");
      String comment;
      comment = "exorciseIndex() would remove reference to this segment";
      msg(infoStream, "    WARNING: " + comment + "; full exception:");
      if (infoStream != null) t.printStackTrace(infoStream);
      msg(infoStream, "");
    } finally {
      if (reader != null) reader.close();
    }
    return segInfoStat;
  }

  /** Tests index sort order. */
  public static Status.IndexSortStatus testSort(
      CodecReader reader, Sort sort, PrintStream infoStream, boolean failFast) throws IOException {
    // This segment claims its documents are sorted according to the incoming sort ... let's make
    // sure:

    long startNS = System.nanoTime();

    Status.IndexSortStatus status = new Status.IndexSortStatus();

    if (sort != null) {
      if (infoStream != null) {
        infoStream.print("    test: index sort..........");
      }

      SortField[] fields = sort.getSort();
      final int[] reverseMul = new int[fields.length];
      final LeafFieldComparator[] comparators = new LeafFieldComparator[fields.length];

      LeafReaderContext readerContext = new LeafReaderContext(reader);

      for (int i = 0; i < fields.length; i++) {
        reverseMul[i] = fields[i].getReverse() ? -1 : 1;
        comparators[i] = fields[i].getComparator(1, Pruning.NONE).getLeafComparator(readerContext);
      }

      try {
        LeafMetaData metaData = reader.getMetaData();
        FieldInfos fieldInfos = reader.getFieldInfos();
        if (metaData.hasBlocks()
            && fieldInfos.getParentField() == null
            && metaData.createdVersionMajor() >= Version.LUCENE_10_0_0.major) {
          throw new IllegalStateException(
              "parent field is not set but the index has document blocks and was created with version: "
                  + metaData.createdVersionMajor());
        }
        final DocIdSetIterator iter;
        if (metaData.hasBlocks() && fieldInfos.getParentField() != null) {
          iter = reader.getNumericDocValues(fieldInfos.getParentField());
        } else {
          iter = DocIdSetIterator.all(reader.maxDoc());
        }
        int prevDoc = iter.nextDoc();
        int nextDoc;
        while ((nextDoc = iter.nextDoc()) != NO_MORE_DOCS) {
          int cmp = 0;
          for (int i = 0; i < comparators.length; i++) {
            // TODO: would be better if copy() didn't cause a term lookup in TermOrdVal & co,
            // the segments are always the same here...
            comparators[i].copy(0, prevDoc);
            comparators[i].setBottom(0);
            cmp = reverseMul[i] * comparators[i].compareBottom(nextDoc);
            if (cmp != 0) {
              break;
            }
          }
          if (cmp > 0) {
            throw new CheckIndexException(
                "segment has indexSort="
                    + sort
                    + " but docID="
                    + (prevDoc)
                    + " sorts after docID="
                    + nextDoc);
          }
          prevDoc = nextDoc;
        }
        msg(
            infoStream,
            String.format(Locale.ROOT, "OK [took %.3f sec]", nsToSec(System.nanoTime() - startNS)));
      } catch (Throwable e) {
        if (failFast) {
          throw IOUtils.rethrowAlways(e);
        }
        msg(infoStream, "ERROR [" + e.getMessage() + "]");
        status.error = e;
        if (infoStream != null) {
          e.printStackTrace(infoStream);
        }
      }
    }

    return status;
  }

  /** Test live docs. */
  public static Status.LiveDocStatus testLiveDocs(
      CodecReader reader, PrintStream infoStream, boolean failFast) throws IOException {
    long startNS = System.nanoTime();
    final Status.LiveDocStatus status = new Status.LiveDocStatus();

    try {
      if (infoStream != null) infoStream.print("    test: check live docs.....");
      final int numDocs = reader.numDocs();
      if (reader.hasDeletions()) {
        Bits liveDocs = reader.getLiveDocs();
        if (liveDocs == null) {
          throw new CheckIndexException("segment should have deletions, but liveDocs is null");
        } else {
          int numLive = 0;
          for (int j = 0; j < liveDocs.length(); j++) {
            if (liveDocs.get(j)) {
              numLive++;
            }
          }
          if (numLive != numDocs) {
            throw new CheckIndexException(
                "liveDocs count mismatch: info=" + numDocs + ", vs bits=" + numLive);
          }
        }

        status.numDeleted = reader.numDeletedDocs();
        msg(
            infoStream,
            String.format(
                Locale.ROOT,
                "OK [%d deleted docs] [took %.3f sec]",
                status.numDeleted,
                nsToSec(System.nanoTime() - startNS)));
      } else {
        Bits liveDocs = reader.getLiveDocs();
        if (liveDocs != null) {
          // it's ok for it to be non-null here, as long as none are set right?
          for (int j = 0; j < liveDocs.length(); j++) {
            if (liveDocs.get(j) == false) {
              throw new CheckIndexException(
                  "liveDocs mismatch: info says no deletions but doc " + j + " is deleted.");
            }
          }
        }
        msg(
            infoStream,
            String.format(
                Locale.ROOT, "OK [took %.3f sec]", (nsToSec(System.nanoTime() - startNS))));
      }

    } catch (Throwable e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR [" + e.getMessage() + "]");
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }

    return status;
  }

  /** Test field infos. */
  public static Status.FieldInfoStatus testFieldInfos(
      CodecReader reader, PrintStream infoStream, boolean failFast) throws IOException {
    long startNS = System.nanoTime();
    final Status.FieldInfoStatus status = new Status.FieldInfoStatus();

    try {
      // Test Field Infos
      if (infoStream != null) {
        infoStream.print("    test: field infos.........");
      }
      FieldInfos fieldInfos = reader.getFieldInfos();
      for (FieldInfo f : fieldInfos) {
        f.checkConsistency();
      }
      msg(
          infoStream,
          String.format(
              Locale.ROOT,
              "OK [%d fields] [took %.3f sec]",
              fieldInfos.size(),
              nsToSec(System.nanoTime() - startNS)));
      status.totFields = fieldInfos.size();
    } catch (Throwable e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR [" + e.getMessage() + "]");
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }

    return status;
  }

  /** Test field norms. */
  public static Status.FieldNormStatus testFieldNorms(
      CodecReader reader, PrintStream infoStream, boolean failFast) throws IOException {
    long startNS = System.nanoTime();
    final Status.FieldNormStatus status = new Status.FieldNormStatus();

    try {
      // Test Field Norms
      if (infoStream != null) {
        infoStream.print("    test: field norms.........");
      }
      NormsProducer normsReader = reader.getNormsReader();
      if (normsReader != null) {
        normsReader = normsReader.getMergeInstance();
      }
      for (FieldInfo info : reader.getFieldInfos()) {
        if (info.hasNorms()) {
          checkNumericDocValues(info.name, normsReader.getNorms(info), normsReader.getNorms(info));
          ++status.totFields;
        }
      }

      msg(
          infoStream,
          String.format(
              Locale.ROOT,
              "OK [%d fields] [took %.3f sec]",
              status.totFields,
              nsToSec(System.nanoTime() - startNS)));
    } catch (Throwable e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR [" + e.getMessage() + "]");
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }

    return status;
  }

  /**
   * checks Fields api is consistent with itself. searcher is optional, to verify with queries. Can
   * be null.
   */
  private static Status.TermIndexStatus checkFields(
      Fields fields,
      Bits liveDocs,
      int maxDoc,
      FieldInfos fieldInfos,
      NormsProducer normsProducer,
      boolean doPrint,
      boolean isVectors,
      PrintStream infoStream,
      boolean verbose,
      int level)
      throws IOException {
    // TODO: we should probably return our own stats thing...?!
    long startNS;
    if (doPrint) {
      startNS = System.nanoTime();
    } else {
      startNS = 0;
    }

    final Status.TermIndexStatus status = new Status.TermIndexStatus();
    int computedFieldCount = 0;

    PostingsEnum postings = null;
    PostingsEnum bulkPostings = null;

    String lastField = null;
    for (String field : fields) {

      // MultiFieldsEnum relies upon this order...
      if (lastField != null && field.compareTo(lastField) <= 0) {
        throw new CheckIndexException(
            "fields out of order: lastField=" + lastField + " field=" + field);
      }
      lastField = field;

      // check that the field is in fieldinfos, and is indexed.
      // TODO: add a separate test to check this for different reader impls
      FieldInfo fieldInfo = fieldInfos.fieldInfo(field);
      if (fieldInfo == null) {
        throw new CheckIndexException(
            "fieldsEnum inconsistent with fieldInfos, no fieldInfos for: " + field);
      }
      if (fieldInfo.getIndexOptions() == IndexOptions.NONE) {
        throw new CheckIndexException(
            "fieldsEnum inconsistent with fieldInfos, isIndexed == false for: " + field);
      }

      // TODO: really the codec should not return a field
      // from FieldsEnum if it has no Terms... but we do
      // this today:
      // assert fields.terms(field) != null;
      computedFieldCount++;

      final Terms terms = fields.terms(field);
      if (terms == null) {
        continue;
      }

      if (terms.getDocCount() > maxDoc) {
        throw new CheckIndexException(
            "docCount > maxDoc for field: "
                + field
                + ", docCount="
                + terms.getDocCount()
                + ", maxDoc="
                + maxDoc);
      }

      final boolean hasFreqs = terms.hasFreqs();
      final boolean hasPositions = terms.hasPositions();
      final boolean hasPayloads = terms.hasPayloads();
      final boolean hasOffsets = terms.hasOffsets();

      BytesRef maxTerm;
      BytesRef minTerm;
      if (isVectors) {
        // Term vectors impls can be very slow for getMax
        maxTerm = null;
        minTerm = null;
      } else {
        BytesRef bb = terms.getMin();
        if (bb != null) {
          assert bb.isValid();
          minTerm = BytesRef.deepCopyOf(bb);
        } else {
          minTerm = null;
        }

        bb = terms.getMax();
        if (bb != null) {
          assert bb.isValid();
          maxTerm = BytesRef.deepCopyOf(bb);
          if (minTerm == null) {
            throw new CheckIndexException(
                "field \"" + field + "\" has null minTerm but non-null maxTerm");
          }
        } else {
          maxTerm = null;
          if (minTerm != null) {
            throw new CheckIndexException(
                "field \"" + field + "\" has non-null minTerm but null maxTerm");
          }
        }
      }

      // term vectors cannot omit TF:
      final boolean expectedHasFreqs =
          (isVectors || fieldInfo.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS) >= 0);

      if (hasFreqs != expectedHasFreqs) {
        throw new CheckIndexException(
            "field \""
                + field
                + "\" should have hasFreqs="
                + expectedHasFreqs
                + " but got "
                + hasFreqs);
      }

      if (isVectors == false) {
        final boolean expectedHasPositions =
            fieldInfo.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) >= 0;
        if (hasPositions != expectedHasPositions) {
          throw new CheckIndexException(
              "field \""
                  + field
                  + "\" should have hasPositions="
                  + expectedHasPositions
                  + " but got "
                  + hasPositions);
        }

        final boolean expectedHasPayloads = fieldInfo.hasPayloads();
        if (hasPayloads != expectedHasPayloads) {
          throw new CheckIndexException(
              "field \""
                  + field
                  + "\" should have hasPayloads="
                  + expectedHasPayloads
                  + " but got "
                  + hasPayloads);
        }

        final boolean expectedHasOffsets =
            fieldInfo
                    .getIndexOptions()
                    .compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)
                >= 0;
        if (hasOffsets != expectedHasOffsets) {
          throw new CheckIndexException(
              "field \""
                  + field
                  + "\" should have hasOffsets="
                  + expectedHasOffsets
                  + " but got "
                  + hasOffsets);
        }
      }

      final TermsEnum termsEnum = terms.iterator();

      boolean hasOrd = true;
      final long termCountStart = status.delTermCount + status.termCount;

      BytesRefBuilder lastTerm = null;

      long sumTotalTermFreq = 0;
      long sumDocFreq = 0;
      FixedBitSet visitedDocs = new FixedBitSet(maxDoc);
      while (true) {

        final BytesRef term = termsEnum.next();
        if (term == null) {
          break;
        }
        // System.out.println("CI: field=" + field + " check term=" + term + " docFreq=" +
        // termsEnum.docFreq());

        assert term.isValid();

        // make sure terms arrive in order according to
        // the comp
        if (lastTerm == null) {
          lastTerm = new BytesRefBuilder();
          lastTerm.copyBytes(term);
        } else {
          if (lastTerm.get().compareTo(term) >= 0) {
            throw new CheckIndexException(
                "terms out of order: lastTerm=" + lastTerm.get() + " term=" + term);
          }
          lastTerm.copyBytes(term);
        }

        if (isVectors == false) {
          if (minTerm == null) {
            // We checked this above:
            assert maxTerm == null;
            throw new CheckIndexException(
                "field=\"" + field + "\": invalid term: term=" + term + ", minTerm=" + minTerm);
          }

          if (term.compareTo(minTerm) < 0) {
            throw new CheckIndexException(
                "field=\"" + field + "\": invalid term: term=" + term + ", minTerm=" + minTerm);
          }

          if (term.compareTo(maxTerm) > 0) {
            throw new CheckIndexException(
                "field=\"" + field + "\": invalid term: term=" + term + ", maxTerm=" + maxTerm);
          }
        }

        final int docFreq = termsEnum.docFreq();
        if (docFreq <= 0) {
          throw new CheckIndexException("docfreq: " + docFreq + " is out of bounds");
        }
        sumDocFreq += docFreq;

        postings = termsEnum.postings(postings, PostingsEnum.ALL);
        bulkPostings = termsEnum.postings(bulkPostings, PostingsEnum.ALL);
        bulkPostings.nextDoc();
        DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
        int bufferIndex = 0;

        if (hasFreqs == false) {
          if (termsEnum.totalTermFreq() != termsEnum.docFreq()) {
            throw new CheckIndexException(
                "field \""
                    + field
                    + "\" hasFreqs is false, but TermsEnum.totalTermFreq()="
                    + termsEnum.totalTermFreq()
                    + " (should be "
                    + termsEnum.docFreq()
                    + ")");
          }
        }

        if (hasOrd) {
          long ord = -1;
          try {
            ord = termsEnum.ord();
          } catch (
              @SuppressWarnings("unused")
              UnsupportedOperationException uoe) {
            hasOrd = false;
          }

          if (hasOrd) {
            final long ordExpected = status.delTermCount + status.termCount - termCountStart;
            if (ord != ordExpected) {
              throw new CheckIndexException(
                  "ord mismatch: TermsEnum has ord=" + ord + " vs actual=" + ordExpected);
            }
          }
        }

        int lastDoc = -1;
        int docCount = 0;
        boolean hasNonDeletedDocs = false;
        long totalTermFreq = 0;
        while (true) {
          final int doc = postings.nextDoc();
          if (doc == DocIdSetIterator.NO_MORE_DOCS) {
            break;
          }
          visitedDocs.set(doc);
          int freq = postings.freq();
          if (freq <= 0) {
            throw new CheckIndexException(
                "term " + term + ": doc " + doc + ": freq " + freq + " is out of bounds");
          }

          if (bufferIndex == buffer.size) {
            bulkPostings.nextPostings(
                (int) Math.min(Integer.MAX_VALUE, bulkPostings.docID() + 64L), buffer);
            bufferIndex = 0;
          }
          if (bufferIndex >= buffer.size) {
            throw new CheckIndexException("Doc " + doc + " not found by PostingsEnum#nextPostings");
          }
          if (doc != buffer.docs[bufferIndex]) {
            throw new CheckIndexException(
                "PostingsEnum#nextPostings returns "
                    + buffer.docs[bufferIndex]
                    + " as next doc while PostingsEnum#nextDoc returns "
                    + doc);
          }
          if (freq != buffer.features[bufferIndex]) {
            throw new CheckIndexException(
                "PostingsEnum#nextPostings returns "
                    + buffer.features[bufferIndex]
                    + " as term freq while PostingsEnum#freq returns "
                    + freq);
          }
          bufferIndex++;

          if (hasFreqs == false) {
            // When a field didn't index freq, it must
            // consistently "lie" and pretend that freq was
            // 1:
            if (postings.freq() != 1) {
              throw new CheckIndexException(
                  "term "
                      + term
                      + ": doc "
                      + doc
                      + ": freq "
                      + freq
                      + " != 1 when Terms.hasFreqs() is false");
            }
          }
          totalTermFreq += freq;

          if (liveDocs == null || liveDocs.get(doc)) {
            hasNonDeletedDocs = true;
            status.totFreq++;
            if (freq >= 0) {
              status.totPos += freq;
            }
          }
          docCount++;

          if (doc <= lastDoc) {
            throw new CheckIndexException(
                "term " + term + ": doc " + doc + " <= lastDoc " + lastDoc);
          }
          if (doc >= maxDoc) {
            throw new CheckIndexException("term " + term + ": doc " + doc + " >= maxDoc " + maxDoc);
          }

          lastDoc = doc;

          int lastPos = -1;
          int lastOffset = 0;
          if (hasPositions) {
            for (int j = 0; j < freq; j++) {
              final int pos = postings.nextPosition();

              if (pos < 0) {
                throw new CheckIndexException(
                    "term " + term + ": doc " + doc + ": pos " + pos + " is out of bounds");
              }
              if (pos > IndexWriter.MAX_POSITION) {
                throw new CheckIndexException(
                    "term "
                        + term
                        + ": doc "
                        + doc
                        + ": pos "
                        + pos
                        + " > IndexWriter.MAX_POSITION="
                        + IndexWriter.MAX_POSITION);
              }
              if (pos < lastPos) {
                throw new CheckIndexException(
                    "term " + term + ": doc " + doc + ": pos " + pos + " < lastPos " + lastPos);
              }
              lastPos = pos;
              BytesRef payload = postings.getPayload();
              if (payload != null) {
                assert payload.isValid();
              }
              if (payload != null && payload.length < 1) {
                throw new CheckIndexException(
                    "term "
                        + term
                        + ": doc "
                        + doc
                        + ": pos "
                        + pos
                        + " payload length is out of bounds "
                        + payload.length);
              }
              if (hasOffsets) {
                int startOffset = postings.startOffset();
                int endOffset = postings.endOffset();
                if (startOffset < 0) {
                  throw new CheckIndexException(
                      "term "
                          + term
                          + ": doc "
                          + doc
                          + ": pos "
                          + pos
                          + ": startOffset "
                          + startOffset
                          + " is out of bounds");
                }
                if (startOffset < lastOffset) {
                  throw new CheckIndexException(
                      "term "
                          + term
                          + ": doc "
                          + doc
                          + ": pos "
                          + pos
                          + ": startOffset "
                          + startOffset
                          + " < lastStartOffset "
                          + lastOffset
                          + "; consider using the FixBrokenOffsets tool in Lucene's backward-codecs module to correct your index");
                }
                if (endOffset < 0) {
                  throw new CheckIndexException(
                      "term "
                          + term
                          + ": doc "
                          + doc
                          + ": pos "
                          + pos
                          + ": endOffset "
                          + endOffset
                          + " is out of bounds");
                }
                if (endOffset < startOffset) {
                  throw new CheckIndexException(
                      "term "
                          + term
                          + ": doc "
                          + doc
                          + ": pos "
                          + pos
                          + ": endOffset "
                          + endOffset
                          + " < startOffset "
                          + startOffset);
                }
                lastOffset = startOffset;
              }
            }
          }
        }

        if (hasNonDeletedDocs) {
          status.termCount++;
        } else {
          status.delTermCount++;
        }

        final long totalTermFreq2 = termsEnum.totalTermFreq();

        if (docCount != docFreq) {
          throw new CheckIndexException(
              "term " + term + " docFreq=" + docFreq + " != tot docs w/o deletions " + docCount);
        }
        if (docFreq > terms.getDocCount()) {
          throw new CheckIndexException(
              "term " + term + " docFreq=" + docFreq + " > docCount=" + terms.getDocCount());
        }
        if (totalTermFreq2 <= 0) {
          throw new CheckIndexException("totalTermFreq: " + totalTermFreq2 + " is out of bounds");
        }
        sumTotalTermFreq += totalTermFreq;
        if (totalTermFreq != totalTermFreq2) {
          throw new CheckIndexException(
              "term "
                  + term
                  + " totalTermFreq="
                  + totalTermFreq2
                  + " != recomputed totalTermFreq="
                  + totalTermFreq);
        }
        if (totalTermFreq2 < docFreq) {
          throw new CheckIndexException(
              "totalTermFreq: " + totalTermFreq2 + " is out of bounds, docFreq=" + docFreq);
        }
        if (hasFreqs == false && totalTermFreq != docFreq) {
          throw new CheckIndexException(
              "term " + term + " totalTermFreq=" + totalTermFreq + " !=  docFreq=" + docFreq);
        }

        // Test skipping
        if (hasPositions) {
          for (int idx = 0; idx < 7; idx++) {
            final int skipDocID = (int) (((idx + 1) * (long) maxDoc) / 8);
            postings = termsEnum.postings(postings, PostingsEnum.ALL);
            final int docID = postings.advance(skipDocID);
            if (docID == DocIdSetIterator.NO_MORE_DOCS) {
              break;
            } else {
              if (docID < skipDocID) {
                throw new CheckIndexException(
                    "term " + term + ": advance(docID=" + skipDocID + ") returned docID=" + docID);
              }
              final int freq = postings.freq();
              if (freq <= 0) {
                throw new CheckIndexException("termFreq " + freq + " is out of bounds");
              }
              int lastPosition = -1;
              int lastOffset = 0;
              for (int posUpto = 0; posUpto < freq; posUpto++) {
                final int pos = postings.nextPosition();

                if (pos < 0) {
                  throw new CheckIndexException("position " + pos + " is out of bounds");
                }
                if (pos < lastPosition) {
                  throw new CheckIndexException(
                      "position " + pos + " is < lastPosition " + lastPosition);
                }
                lastPosition = pos;
                if (hasOffsets) {
                  int startOffset = postings.startOffset();
                  int endOffset = postings.endOffset();
                  // NOTE: we cannot enforce any bounds whatsoever on vectors... they were a
                  // free-for-all before?
                  // but for offsets in the postings lists these checks are fine: they were always
                  // enforced by IndexWriter
                  if (isVectors == false) {
                    if (startOffset < 0) {
                      throw new CheckIndexException(
                          "term "
                              + term
                              + ": doc "
                              + docID
                              + ": pos "
                              + pos
                              + ": startOffset "
                              + startOffset
                              + " is out of bounds");
                    }
                    if (startOffset < lastOffset) {
                      throw new CheckIndexException(
                          "term "
                              + term
                              + ": doc "
                              + docID
                              + ": pos "
                              + pos
                              + ": startOffset "
                              + startOffset
                              + " < lastStartOffset "
                              + lastOffset);
                    }
                    if (endOffset < 0) {
                      throw new CheckIndexException(
                          "term "
                              + term
                              + ": doc "
                              + docID
                              + ": pos "
                              + pos
                              + ": endOffset "
                              + endOffset
                              + " is out of bounds");
                    }
                    if (endOffset < startOffset) {
                      throw new CheckIndexException(
                          "term "
                              + term
                              + ": doc "
                              + docID
                              + ": pos "
                              + pos
                              + ": endOffset "
                              + endOffset
                              + " < startOffset "
                              + startOffset);
                    }
                  }
                  lastOffset = startOffset;
                }
              }

              final int nextDocID = postings.nextDoc();
              if (nextDocID == DocIdSetIterator.NO_MORE_DOCS) {
                break;
              }
              if (nextDocID <= docID) {
                throw new CheckIndexException(
                    "term "
                        + term
                        + ": advance(docID="
                        + skipDocID
                        + "), then .next() returned docID="
                        + nextDocID
                        + " vs prev docID="
                        + docID);
              }
            }

            if (isVectors) {
              // Only 1 doc in the postings for term vectors, so we only test 1 advance:
              break;
            }
          }
        } else {
          for (int idx = 0; idx < 7; idx++) {
            final int skipDocID = (int) (((idx + 1) * (long) maxDoc) / 8);
            postings = termsEnum.postings(postings, PostingsEnum.NONE);
            final int docID = postings.advance(skipDocID);
            if (docID == DocIdSetIterator.NO_MORE_DOCS) {
              break;
            } else {
              if (docID < skipDocID) {
                throw new CheckIndexException(
                    "term " + term + ": advance(docID=" + skipDocID + ") returned docID=" + docID);
              }
              final int nextDocID = postings.nextDoc();
              if (nextDocID == DocIdSetIterator.NO_MORE_DOCS) {
                break;
              }
              if (nextDocID <= docID) {
                throw new CheckIndexException(
                    "term "
                        + term
                        + ": advance(docID="
                        + skipDocID
                        + "), then .next() returned docID="
                        + nextDocID
                        + " vs prev docID="
                        + docID);
              }
            }
            if (isVectors) {
              // Only 1 doc in the postings for term vectors, so we only test 1 advance:
              break;
            }
          }
        }

        // Checking score blocks and doc ID runs is heavy, we only do it on long postings lists, on
        // every 1024th term or if slow checks are enabled.
        if (level >= Level.MIN_LEVEL_FOR_SLOW_CHECKS
            || docFreq > 1024
            || (status.termCount + status.delTermCount) % 1024 == 0) {
          postings = termsEnum.postings(postings, PostingsEnum.NONE);
          checkDocIDRuns(postings);
          if (hasFreqs) {
            postings = termsEnum.postings(postings, PostingsEnum.FREQS);
            checkDocIDRuns(postings);
          }
          if (hasPositions) {
            postings = termsEnum.postings(postings, PostingsEnum.POSITIONS);
            checkDocIDRuns(postings);
          }

          // First check max scores and block uptos
          // But only if slow checks are enabled since we visit all docs
          if (level >= Level.MIN_LEVEL_FOR_SLOW_CHECKS) {
            int max = -1;
            int maxFreq = 0;
            ImpactsEnum impactsEnum = termsEnum.impacts(PostingsEnum.FREQS);
            postings = termsEnum.postings(postings, PostingsEnum.FREQS);
            for (int doc = impactsEnum.nextDoc(); ; doc = impactsEnum.nextDoc()) {
              if (postings.nextDoc() != doc) {
                throw new CheckIndexException(
                    "Wrong next doc: " + doc + ", expected " + postings.docID());
              }
              if (doc == DocIdSetIterator.NO_MORE_DOCS) {
                break;
              }
              if (postings.freq() != impactsEnum.freq()) {
                throw new CheckIndexException(
                    "Wrong freq, expected " + postings.freq() + ", but got " + impactsEnum.freq());
              }
              if (doc > max) {
                impactsEnum.advanceShallow(doc);
                Impacts impacts = impactsEnum.getImpacts();
                checkImpacts(impacts, doc);
                max = impacts.getDocIdUpTo(0);
                List<Impact> impacts0 = impacts.getImpacts(0);
                maxFreq = impacts0.get(impacts0.size() - 1).freq;
              }
              if (impactsEnum.freq() > maxFreq) {
                throw new CheckIndexException(
                    "freq "
                        + impactsEnum.freq()
                        + " is greater than the max freq according to impacts "
                        + maxFreq);
              }
            }
          }

          // Now check advancing
          ImpactsEnum impactsEnum = termsEnum.impacts(PostingsEnum.FREQS);
          postings = termsEnum.postings(postings, PostingsEnum.FREQS);

          int max = -1;
          int maxFreq = 0;
          while (true) {
            int doc = impactsEnum.docID();
            boolean advance;
            int target;
            if (((field.hashCode() + doc) & 1) == 1) {
              advance = false;
              target = doc + 1;
            } else {
              advance = true;
              int delta =
                  Math.min(
                      1 + ((31 * field.hashCode() + doc) & 0x1ff),
                      DocIdSetIterator.NO_MORE_DOCS - doc);
              target = impactsEnum.docID() + delta;
            }

            if (target > max && target % 2 == 1) {
              int delta =
                  Math.min(
                      (31 * field.hashCode() + target) & 0x1ff,
                      DocIdSetIterator.NO_MORE_DOCS - target);
              max = target + delta;
              impactsEnum.advanceShallow(target);
              Impacts impacts = impactsEnum.getImpacts();
              checkImpacts(impacts, doc);
              maxFreq = Integer.MAX_VALUE;
              for (int impactsLevel = 0; impactsLevel < impacts.numLevels(); ++impactsLevel) {
                if (impacts.getDocIdUpTo(impactsLevel) >= max) {
                  List<Impact> perLevelImpacts = impacts.getImpacts(impactsLevel);
                  maxFreq = perLevelImpacts.get(perLevelImpacts.size() - 1).freq;
                  break;
                }
              }
            }

            if (advance) {
              doc = impactsEnum.advance(target);
            } else {
              doc = impactsEnum.nextDoc();
            }

            if (postings.advance(target) != doc) {
              throw new CheckIndexException(
                  "Impacts do not advance to the same document as postings for target "
                      + target
                      + ", postings: "
                      + postings.docID()
                      + ", impacts: "
                      + doc);
            }
            if (doc == DocIdSetIterator.NO_MORE_DOCS) {
              break;
            }
            if (postings.freq() != impactsEnum.freq()) {
              throw new CheckIndexException(
                  "Wrong freq, expected " + postings.freq() + ", but got " + impactsEnum.freq());
            }

            if (doc >= max) {
              int delta =
                  Math.min(
                      (31 * field.hashCode() + target & 0x1ff),
                      DocIdSetIterator.NO_MORE_DOCS - doc);
              max = doc + delta;
              impactsEnum.advanceShallow(doc);
              Impacts impacts = impactsEnum.getImpacts();
              checkImpacts(impacts, doc);
              maxFreq = Integer.MAX_VALUE;
              for (int impactsLevel = 0; impactsLevel < impacts.numLevels(); ++impactsLevel) {
                if (impacts.getDocIdUpTo(impactsLevel) >= max) {
                  List<Impact> perLevelImpacts = impacts.getImpacts(impactsLevel);
                  maxFreq = perLevelImpacts.get(perLevelImpacts.size() - 1).freq;
                  break;
                }
              }
            }

            if (impactsEnum.freq() > maxFreq) {
              throw new CheckIndexException(
                  "Term frequency "
                      + impactsEnum.freq()
                      + " is greater than the max freq according to impacts "
                      + maxFreq);
            }
          }
        }
      }

      if (minTerm != null && status.termCount + status.delTermCount == 0) {
        throw new CheckIndexException(
            "field=\"" + field + "\": minTerm is non-null yet we saw no terms: " + minTerm);
      }

      final Terms fieldTerms = fields.terms(field);
      if (fieldTerms == null) {
        // Unusual: the FieldsEnum returned a field but
        // the Terms for that field is null; this should
        // only happen if it's a ghost field (field with
        // no terms, e.g. there used to be terms but all
        // docs got deleted and then merged away):

      } else {

        long fieldTermCount = (status.delTermCount + status.termCount) - termCountStart;

        final Object stats = fieldTerms.getStats();
        assert stats != null;
        if (status.blockTreeStats == null) {
          status.blockTreeStats = new HashMap<>();
        }
        status.blockTreeStats.put(field, stats);

        final long actualSumDocFreq = fields.terms(field).getSumDocFreq();
        if (sumDocFreq != actualSumDocFreq) {
          throw new CheckIndexException(
              "sumDocFreq for field "
                  + field
                  + "="
                  + actualSumDocFreq
                  + " != recomputed sumDocFreq="
                  + sumDocFreq);
        }

        final long actualSumTotalTermFreq = fields.terms(field).getSumTotalTermFreq();
        if (sumTotalTermFreq != actualSumTotalTermFreq) {
          throw new CheckIndexException(
              "sumTotalTermFreq for field "
                  + field
                  + "="
                  + actualSumTotalTermFreq
                  + " != recomputed sumTotalTermFreq="
                  + sumTotalTermFreq);
        }

        if (hasFreqs == false && sumTotalTermFreq != sumDocFreq) {
          throw new CheckIndexException(
              "sumTotalTermFreq for field "
                  + field
                  + " should be "
                  + sumDocFreq
                  + ", got sumTotalTermFreq="
                  + sumTotalTermFreq);
        }

        final int v = fieldTerms.getDocCount();
        if (visitedDocs.cardinality() != v) {
          throw new CheckIndexException(
              "docCount for field "
                  + field
                  + "="
                  + v
                  + " != recomputed docCount="
                  + visitedDocs.cardinality());
        }

        if (fieldInfo.hasNorms() && isVectors == false) {
          final NumericDocValues norms = normsProducer.getNorms(fieldInfo);
          // count of valid norm values found for the field
          int actualCount = 0;
          // Cross-check terms with norms
          for (int doc = norms.nextDoc();
              doc != DocIdSetIterator.NO_MORE_DOCS;
              doc = norms.nextDoc()) {
            if (liveDocs != null && liveDocs.get(doc) == false) {
              // Norms may only be out of sync with terms on deleted documents.
              // This happens when a document fails indexing and in that case it
              // should be immediately marked as deleted by the IndexWriter.
              continue;
            }
            final long norm = norms.longValue();
            if (norm != 0) {
              actualCount++;
              if (visitedDocs.get(doc) == false) {
                throw new CheckIndexException(
                    "Document "
                        + doc
                        + " doesn't have terms according to postings but has a norm value that is not zero: "
                        + Long.toUnsignedString(norm));
              }
            } else if (visitedDocs.get(doc)) {
              throw new CheckIndexException(
                  "Document "
                      + doc
                      + " has terms according to postings but its norm value is 0, which may only be used on documents that have no terms");
            }
          }
          int expectedCount = 0;
          for (int doc = visitedDocs.nextSetBit(0);
              doc != DocIdSetIterator.NO_MORE_DOCS;
              doc =
                  doc + 1 >= visitedDocs.length()
                      ? DocIdSetIterator.NO_MORE_DOCS
                      : visitedDocs.nextSetBit(doc + 1)) {
            if (liveDocs != null && liveDocs.get(doc) == false) {
              // Norms may only be out of sync with terms on deleted documents.
              // This happens when a document fails indexing and in that case it
              // should be immediately marked as deleted by the IndexWriter.
              continue;
            }
            expectedCount++;
          }
          if (expectedCount != actualCount) {
            throw new CheckIndexException(
                "actual norm count: " + actualCount + " but expected: " + expectedCount);
          }
        }

        // Test seek to last term:
        if (lastTerm != null) {
          if (termsEnum.seekCeil(lastTerm.get()) != TermsEnum.SeekStatus.FOUND) {
            throw new CheckIndexException("seek to last term " + lastTerm.get() + " failed");
          }
          if (termsEnum.term().equals(lastTerm.get()) == false) {
            throw new CheckIndexException(
                "seek to last term "
                    + lastTerm.get()
                    + " returned FOUND but seeked to the wrong term "
                    + termsEnum.term());
          }

          int expectedDocFreq = termsEnum.docFreq();
          PostingsEnum d = termsEnum.postings(null, PostingsEnum.NONE);
          int docFreq = 0;
          while (d.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            docFreq++;
          }
          if (docFreq != expectedDocFreq) {
            throw new CheckIndexException(
                "docFreq for last term "
                    + lastTerm.get()
                    + "="
                    + expectedDocFreq
                    + " != recomputed docFreq="
                    + docFreq);
          }
        }

        // check unique term count
        long termCount = -1;

        if (fieldTermCount > 0) {
          termCount = fields.terms(field).size();

          if (termCount != -1 && termCount != fieldTermCount) {
            throw new CheckIndexException(
                "termCount mismatch " + termCount + " vs " + fieldTermCount);
          }
        }

        // Test seeking by ord
        if (hasOrd && status.termCount - termCountStart > 0) {
          int seekCount = (int) Math.min(10000L, termCount);
          if (seekCount > 0) {
            BytesRef[] seekTerms = new BytesRef[seekCount];

            // Seek by ord
            for (int i = seekCount - 1; i >= 0; i--) {
              long ord = i * (termCount / seekCount);
              termsEnum.seekExact(ord);
              long actualOrd = termsEnum.ord();
              if (actualOrd != ord) {
                throw new CheckIndexException("seek to ord " + ord + " returned ord " + actualOrd);
              }
              seekTerms[i] = BytesRef.deepCopyOf(termsEnum.term());
            }

            // Seek by term
            for (int i = seekCount - 1; i >= 0; i--) {
              if (termsEnum.seekCeil(seekTerms[i]) != TermsEnum.SeekStatus.FOUND) {
                throw new CheckIndexException("seek to existing term " + seekTerms[i] + " failed");
              }
              if (termsEnum.term().equals(seekTerms[i]) == false) {
                throw new CheckIndexException(
                    "seek to existing term "
                        + seekTerms[i]
                        + " returned FOUND but seeked to the wrong term "
                        + termsEnum.term());
              }

              postings = termsEnum.postings(postings, PostingsEnum.NONE);
              if (postings == null) {
                throw new CheckIndexException(
                    "null DocsEnum from to existing term " + seekTerms[i]);
              }
            }
          }
        }

        // Test Terms#intersect
        // An automaton that should match a good number of terms
        Automaton automaton =
            Operations.concatenate(
                Arrays.asList(
                    Automata.makeAnyBinary(),
                    Automata.makeCharRange('a', 'e'),
                    Automata.makeAnyBinary()));
        BytesRef startTerm = null;
        checkTermsIntersect(terms, automaton, startTerm);

        startTerm = new BytesRef();
        checkTermsIntersect(terms, automaton, startTerm);

        automaton = Automata.makeNonEmptyBinary();
        startTerm = new BytesRef(new byte[] {'l'});
        checkTermsIntersect(terms, automaton, startTerm);

        // a term that likely compares greater than every other term in the dictionary
        startTerm = new BytesRef(new byte[] {(byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF});
        checkTermsIntersect(terms, automaton, startTerm);
      }
    }

    int fieldCount = fields.size();

    if (fieldCount != -1) {
      if (fieldCount < 0) {
        throw new CheckIndexException("invalid fieldCount: " + fieldCount);
      }
      if (fieldCount != computedFieldCount) {
        throw new CheckIndexException(
            "fieldCount mismatch "
                + fieldCount
                + " vs recomputed field count "
                + computedFieldCount);
      }
    }

    if (doPrint) {
      msg(
          infoStream,
          String.format(
              Locale.ROOT,
              "OK [%d terms; %d terms/docs pairs; %d tokens] [took %.3f sec]",
              status.termCount,
              status.totFreq,
              status.totPos,
              nsToSec(System.nanoTime() - startNS)));
    }

    if (verbose && status.blockTreeStats != null && infoStream != null && status.termCount > 0) {
      for (Map.Entry<String, Object> ent : status.blockTreeStats.entrySet()) {
        infoStream.println("      field \"" + ent.getKey() + "\":");
        infoStream.println("      " + ent.getValue().toString().replace("\n", "\n      "));
      }
    }

    return status;
  }

  private static void checkTermsIntersect(Terms terms, Automaton automaton, BytesRef startTerm)
      throws IOException {
    TermsEnum allTerms = terms.iterator();
    automaton = Operations.determinize(automaton, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT);
    CompiledAutomaton compiledAutomaton = new CompiledAutomaton(automaton, false, true, true);
    ByteRunAutomaton runAutomaton = new ByteRunAutomaton(automaton, true);
    TermsEnum filteredTerms = terms.intersect(compiledAutomaton, startTerm);
    BytesRef term;
    if (startTerm != null) {
      switch (allTerms.seekCeil(startTerm)) {
        case FOUND:
          term = allTerms.next();
          break;
        case NOT_FOUND:
          term = allTerms.term();
          break;
        case END:
        default:
          term = null;
          break;
      }
    } else {
      term = allTerms.next();
    }
    for (; term != null; term = allTerms.next()) {
      if (runAutomaton.run(term.bytes, term.offset, term.length)) {
        BytesRef filteredTerm = filteredTerms.next();
        if (Objects.equals(term, filteredTerm) == false) {
          throw new CheckIndexException(
              "Expected next filtered term: " + term + ", but got " + filteredTerm);
        }
      }
    }
    BytesRef filteredTerm = filteredTerms.next();
    if (filteredTerm != null) {
      throw new CheckIndexException("Expected exhausted TermsEnum, but got " + filteredTerm);
    }
  }

  private static void checkDocIDRuns(DocIdSetIterator iterator) throws IOException {
    int prevDoc = -1;
    int runEnd = 0;
    for (int doc = iterator.nextDoc();
        doc != DocIdSetIterator.NO_MORE_DOCS;
        doc = iterator.nextDoc()) {
      if (prevDoc + 1 < runEnd && doc != prevDoc + 1) {
        throw new CheckIndexException(
            "Run end is " + runEnd + " but next doc after " + prevDoc + " is " + doc);
      }
      int newRunEnd = iterator.docIDRunEnd();
      if (newRunEnd <= doc) {
        throw new CheckIndexException("Run end " + newRunEnd + " is <= doc ID " + doc);
      }
      if (newRunEnd > runEnd) {
        runEnd = newRunEnd;
      }
      prevDoc = doc;
    }

    if (runEnd != prevDoc + 1) {
      throw new CheckIndexException("Run end is " + runEnd + " but last doc is " + prevDoc);
    }
  }

  /**
   * For use in tests only.
   *
   * @lucene.internal
   */
  static void checkImpacts(Impacts impacts, int lastTarget) {
    final int numLevels = impacts.numLevels();
    if (numLevels < 1) {
      throw new CheckIndexException("The number of impact levels must be >= 1, got " + numLevels);
    }

    int docIdUpTo0 = impacts.getDocIdUpTo(0);
    if (docIdUpTo0 < lastTarget) {
      throw new CheckIndexException(
          "getDocIdUpTo returned "
              + docIdUpTo0
              + " on level 0, which is less than the target "
              + lastTarget);
    }

    for (int impactsLevel = 1; impactsLevel < numLevels; ++impactsLevel) {
      int docIdUpTo = impacts.getDocIdUpTo(impactsLevel);
      int previousDocIdUpTo = impacts.getDocIdUpTo(impactsLevel - 1);
      if (docIdUpTo < previousDocIdUpTo) {
        throw new CheckIndexException(
            "Decreasing return for getDocIdUpTo: level "
                + (impactsLevel - 1)
                + " returned "
                + previousDocIdUpTo
                + " but level "
                + impactsLevel
                + " returned "
                + docIdUpTo
                + " for target "
                + lastTarget);
      }
    }

    for (int impactsLevel = 0; impactsLevel < numLevels; ++impactsLevel) {
      List<Impact> perLevelImpacts = impacts.getImpacts(impactsLevel);
      if (perLevelImpacts.isEmpty()) {
        throw new CheckIndexException("Got empty list of impacts on level " + impactsLevel);
      }
      Impact first = perLevelImpacts.get(0);
      if (first.freq < 1) {
        throw new CheckIndexException("First impact had a freq <= 0: " + first);
      }
      if (first.norm == 0) {
        throw new CheckIndexException("First impact had a norm == 0: " + first);
      }
      // Impacts must be in increasing order of norm AND freq
      Impact previous = first;
      for (int i = 1; i < perLevelImpacts.size(); ++i) {
        Impact impact = perLevelImpacts.get(i);
        if (impact.freq <= previous.freq || Long.compareUnsigned(impact.norm, previous.norm) <= 0) {
          throw new CheckIndexException(
              "Impacts are not ordered or contain dups, got " + previous + " then " + impact);
        }
      }
      if (impactsLevel > 0) {
        // Make sure that impacts at level N trigger better scores than an impactsLevel N-1
        Iterator<Impact> previousIt = impacts.getImpacts(impactsLevel - 1).iterator();
        previous = previousIt.next();
        Iterator<Impact> it = perLevelImpacts.iterator();
        Impact impact = it.next();
        while (previousIt.hasNext()) {
          previous = previousIt.next();
          if (previous.freq <= impact.freq
              && Long.compareUnsigned(previous.norm, impact.norm) >= 0) {
            // previous triggers a lower score than the current impact, all good
            continue;
          }
          if (it.hasNext() == false) {
            throw new CheckIndexException(
                "Found impact "
                    + previous
                    + " on level "
                    + (impactsLevel - 1)
                    + " but no impact on level "
                    + impactsLevel
                    + " triggers a better score: "
                    + perLevelImpacts);
          }
          impact = it.next();
        }
      }
    }
  }

  /** Test the term index. */
  public static Status.TermIndexStatus testPostings(CodecReader reader, PrintStream infoStream)
      throws IOException {
    return testPostings(reader, infoStream, false, Level.MIN_LEVEL_FOR_SLOW_CHECKS, false);
  }

  /** Test the term index. */
  public static Status.TermIndexStatus testPostings(
      CodecReader reader, PrintStream infoStream, boolean verbose, int level, boolean failFast)
      throws IOException {

    // TODO: we should go and verify term vectors match, if the Level is high enough to
    // include slow checks
    Status.TermIndexStatus status;
    final int maxDoc = reader.maxDoc();

    try {
      if (infoStream != null) {
        infoStream.print("    test: terms, freq, prox...");
      }

      FieldsProducer fields = reader.getPostingsReader();
      if (fields != null) {
        fields = fields.getMergeInstance();
      } else {
        return new Status.TermIndexStatus();
      }
      final FieldInfos fieldInfos = reader.getFieldInfos();
      NormsProducer normsProducer = reader.getNormsReader();
      if (normsProducer != null) {
        normsProducer = normsProducer.getMergeInstance();
      }
      status =
          checkFields(
              fields,
              reader.getLiveDocs(),
              maxDoc,
              fieldInfos,
              normsProducer,
              true,
              false,
              infoStream,
              verbose,
              level);
    } catch (Throwable e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR: " + e);
      status = new Status.TermIndexStatus();
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }

    return status;
  }

  /** Test the points index. */
  public static Status.PointsStatus testPoints(
      CodecReader reader, PrintStream infoStream, boolean failFast) throws IOException {
    if (infoStream != null) {
      infoStream.print("    test: points..............");
    }
    long startNS = System.nanoTime();
    FieldInfos fieldInfos = reader.getFieldInfos();
    Status.PointsStatus status = new Status.PointsStatus();
    try {

      if (fieldInfos.hasPointValues()) {
        PointsReader pointsReader = reader.getPointsReader();
        if (pointsReader == null) {
          throw new CheckIndexException(
              "there are fields with points, but reader.getPointsReader() is null");
        }
        for (FieldInfo fieldInfo : fieldInfos) {
          if (fieldInfo.getPointDimensionCount() > 0) {
            PointValues values = pointsReader.getValues(fieldInfo.name);
            if (values == null) {
              continue;
            }

            status.totalValueFields++;

            long size = values.size();
            int docCount = values.getDocCount();

            final long crossCost =
                values.estimatePointCount(
                    new ConstantRelationIntersectVisitor(Relation.CELL_CROSSES_QUERY));
            if (crossCost < size / 2) {
              throw new CheckIndexException(
                  "estimatePointCount should return >= size/2 when all cells match");
            }
            final long insideCost =
                values.estimatePointCount(
                    new ConstantRelationIntersectVisitor(Relation.CELL_INSIDE_QUERY));
            if (insideCost < size) {
              throw new CheckIndexException(
                  "estimatePointCount should return >= size when all cells fully match");
            }
            final long outsideCost =
                values.estimatePointCount(
                    new ConstantRelationIntersectVisitor(Relation.CELL_OUTSIDE_QUERY));
            if (outsideCost != 0) {
              throw new CheckIndexException(
                  "estimatePointCount should return 0 when no cells match");
            }

            VerifyPointsVisitor visitor =
                new VerifyPointsVisitor(fieldInfo.name, reader.maxDoc(), values);
            values.intersect(visitor);

            if (visitor.getPointCountSeen() != size) {
              throw new CheckIndexException(
                  "point values for field \""
                      + fieldInfo.name
                      + "\" claims to have size="
                      + size
                      + " points, but in fact has "
                      + visitor.getPointCountSeen());
            }

            if (visitor.getDocCountSeen() != docCount) {
              throw new CheckIndexException(
                  "point values for field \""
                      + fieldInfo.name
                      + "\" claims to have docCount="
                      + docCount
                      + " but in fact has "
                      + visitor.getDocCountSeen());
            }

            status.totalValuePoints += visitor.getPointCountSeen();
          }
        }
      }

      msg(
          infoStream,
          String.format(
              Locale.ROOT,
              "OK [%d fields, %d points] [took %.3f sec]",
              status.totalValueFields,
              status.totalValuePoints,
              nsToSec(System.nanoTime() - startNS)));

    } catch (Throwable e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR: " + e);
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }

    return status;
  }

  /** Test the vectors index. */
  public static Status.VectorValuesStatus testVectors(
      CodecReader reader, PrintStream infoStream, boolean failFast) throws IOException {
    if (infoStream != null) {
      infoStream.print("    test: vectors.............");
    }
    long startNS = System.nanoTime();
    FieldInfos fieldInfos = reader.getFieldInfos();
    Status.VectorValuesStatus status = new Status.VectorValuesStatus();
    try {

      if (fieldInfos.hasVectorValues()) {
        for (FieldInfo fieldInfo : fieldInfos) {
          if (fieldInfo.hasVectorValues()) {
            int dimension = fieldInfo.getVectorDimension();
            if (dimension <= 0) {
              throw new CheckIndexException(
                  "Field \""
                      + fieldInfo.name
                      + "\" has vector values but dimension is "
                      + dimension);
            }
            if (reader.getFloatVectorValues(fieldInfo.name) == null
                && reader.getByteVectorValues(fieldInfo.name) == null) {
              continue;
            }

            status.totalKnnVectorFields++;
            switch (fieldInfo.getVectorEncoding()) {
              case BYTE:
                checkByteVectorValues(
                    Objects.requireNonNull(reader.getByteVectorValues(fieldInfo.name)),
                    fieldInfo,
                    status,
                    reader);
                break;
              case FLOAT32:
                checkFloatVectorValues(
                    Objects.requireNonNull(reader.getFloatVectorValues(fieldInfo.name)),
                    fieldInfo,
                    status,
                    reader);
                break;
              default:
                throw new CheckIndexException(
                    "Field \""
                        + fieldInfo.name
                        + "\" has unexpected vector encoding: "
                        + fieldInfo.getVectorEncoding());
            }
          }
        }
      }
      msg(
          infoStream,
          String.format(
              Locale.ROOT,
              "OK [%d fields, %d vectors] [took %.3f sec]",
              status.totalKnnVectorFields,
              status.totalVectorValues,
              nsToSec(System.nanoTime() - startNS)));

    } catch (Throwable e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR: " + e);
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }

    return status;
  }

  /** Test the HNSW graph. */
  public static Status.HnswGraphsStatus testHnswGraphs(
      CodecReader reader, PrintStream infoStream, boolean failFast) throws IOException {
    if (infoStream != null) {
      infoStream.print("    test: hnsw graphs.........");
    }
    long startNS = System.nanoTime();
    Status.HnswGraphsStatus status = new Status.HnswGraphsStatus();
    KnnVectorsReader vectorsReader = reader.getVectorReader();
    FieldInfos fieldInfos = reader.getFieldInfos();

    try {
      if (fieldInfos.hasVectorValues()) {
        for (FieldInfo fieldInfo : fieldInfos) {
          if (fieldInfo.hasVectorValues()) {
            KnnVectorsReader fieldReader = getFieldReaderForName(vectorsReader, fieldInfo.name);
            if (fieldReader instanceof HnswGraphProvider graphProvider) {
              HnswGraph hnswGraph = graphProvider.getGraph(fieldInfo.name);
              testHnswGraph(hnswGraph, fieldInfo.name, status);
            }
          }
        }
      }
      msg(
          infoStream,
          String.format(
              Locale.ROOT,
              "OK [%d fields] [took %.3f sec]",
              status.hnswGraphsStatusByField.size(),
              nsToSec(System.nanoTime() - startNS)));
      printHnswInfo(infoStream, status.hnswGraphsStatusByField);
    } catch (Exception e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR: " + e);
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }

    return status;
  }

  private static KnnVectorsReader getFieldReaderForName(
      KnnVectorsReader vectorsReader, String fieldName) {
    if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      return fieldsReader.getFieldReader(fieldName);
    } else {
      return vectorsReader;
    }
  }

  private static void printHnswInfo(
      PrintStream infoStream, Map<String, CheckIndex.Status.HnswGraphStatus> fieldsStatus) {
    for (Map.Entry<String, CheckIndex.Status.HnswGraphStatus> entry : fieldsStatus.entrySet()) {
      String fieldName = entry.getKey();
      CheckIndex.Status.HnswGraphStatus status = entry.getValue();
      msg(infoStream, "      hnsw field name: " + fieldName);

      int numLevels = Math.min(status.numNodesAtLevel.size(), status.connectednessAtLevel.size());
      for (int level = numLevels - 1; level >= 0; level--) {
        int numNodes = status.numNodesAtLevel.get(level);
        String connectedness = status.connectednessAtLevel.get(level);
        msg(
            infoStream,
            String.format(
                Locale.ROOT,
                "        level %d: %d nodes, %s connected",
                level,
                numNodes,
                connectedness));
      }
    }
  }

  private static void testHnswGraph(
      HnswGraph hnswGraph, String fieldName, Status.HnswGraphsStatus status)
      throws IOException, CheckIndexException {
    if (hnswGraph != null) {
      status.hnswGraphsStatusByField.put(fieldName, new Status.HnswGraphStatus());
      final int numLevels = hnswGraph.numLevels();
      status.hnswGraphsStatusByField.get(fieldName).numNodesAtLevel =
          new ArrayList<>(Collections.nCopies(numLevels, null));
      status.hnswGraphsStatusByField.get(fieldName).connectednessAtLevel =
          new ArrayList<>(Collections.nCopies(numLevels, null));
      // Perform checks on each level of the HNSW graph
      for (int level = numLevels - 1; level >= 0; level--) {
        // Collect BitSet of all nodes on this level
        BitSet nodesOnThisLevel = new FixedBitSet(hnswGraph.size());
        HnswGraph.NodesIterator nodesIterator = hnswGraph.getNodesOnLevel(level);
        while (nodesIterator.hasNext()) {
          nodesOnThisLevel.set(nodesIterator.nextInt());
        }

        nodesIterator = hnswGraph.getNodesOnLevel(level);
        // Perform checks on each node on the level
        while (nodesIterator.hasNext()) {
          int node = nodesIterator.nextInt();
          if (node < 0 || node > hnswGraph.size() - 1) {
            throw new CheckIndexException(
                "Field \""
                    + fieldName
                    + "\" has node: "
                    + node
                    + " not in the expected range [0, "
                    + (hnswGraph.size() - 1)
                    + "]");
          }

          // Perform checks on the node's neighbors
          hnswGraph.seek(level, node);
          int nbr, lastNeighbor = -1, firstNeighbor = -1;
          while ((nbr = hnswGraph.nextNeighbor()) != NO_MORE_DOCS) {
            if (!nodesOnThisLevel.get(nbr)) {
              throw new CheckIndexException(
                  "Field \""
                      + fieldName
                      + "\" has node: "
                      + node
                      + " with a neighbor "
                      + nbr
                      + " which is not on its level ("
                      + level
                      + ")");
            }
            if (firstNeighbor == -1) {
              firstNeighbor = nbr;
            }
            if (nbr < lastNeighbor) {
              throw new CheckIndexException(
                  "Field \""
                      + fieldName
                      + "\" has neighbors out of order for node "
                      + node
                      + ": "
                      + nbr
                      + "<"
                      + lastNeighbor
                      + " 1st="
                      + firstNeighbor);
            } else if (nbr == lastNeighbor) {
              throw new CheckIndexException(
                  "Field \""
                      + fieldName
                      + "\" has repeated neighbors of node "
                      + node
                      + " with value "
                      + nbr);
            }
            lastNeighbor = nbr;
          }
        }
        int numNodesOnLayer = nodesIterator.size();
        status.hnswGraphsStatusByField.get(fieldName).numNodesAtLevel.set(level, numNodesOnLayer);

        // Evaluate connectedness at this level by measuring the number of nodes reachable from the
        // entry point
        IntIntHashMap connectedNodes = getConnectedNodesOnLevel(hnswGraph, numNodesOnLayer, level);
        status
            .hnswGraphsStatusByField
            .get(fieldName)
            .connectednessAtLevel
            .set(level, connectedNodes.size() + "/" + numNodesOnLayer);
      }
    }
  }

  private static IntIntHashMap getConnectedNodesOnLevel(
      HnswGraph hnswGraph, int numNodesOnLayer, int level) throws IOException {
    IntIntHashMap connectedNodes = new IntIntHashMap(numNodesOnLayer);
    int entryPoint = hnswGraph.entryNode();
    Deque<Integer> stack = new ArrayDeque<>();
    stack.push(entryPoint);
    while (!stack.isEmpty()) {
      int node = stack.pop();
      if (connectedNodes.containsKey(node)) {
        continue;
      }
      connectedNodes.put(node, 1);
      hnswGraph.seek(level, node);
      int friendOrd;
      while ((friendOrd = hnswGraph.nextNeighbor()) != NO_MORE_DOCS) {
        stack.push(friendOrd);
      }
    }
    return connectedNodes;
  }

  private static boolean vectorsReaderSupportsSearch(CodecReader codecReader, String fieldName) {
    KnnVectorsReader vectorsReader = codecReader.getVectorReader();
    if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader perFieldReader) {
      vectorsReader = perFieldReader.getFieldReader(fieldName);
    }
    return (vectorsReader instanceof FlatVectorsReader) == false;
  }

  private static void checkFloatVectorValues(
      FloatVectorValues values,
      FieldInfo fieldInfo,
      CheckIndex.Status.VectorValuesStatus status,
      CodecReader codecReader)
      throws IOException {
    int count = 0;
    int everyNdoc = Math.max(values.size() / 64, 1);
    while (count < values.size()) {
      // search the first maxNumSearches vectors to exercise the graph
      if (values.ordToDoc(count) % everyNdoc == 0) {
        KnnCollector collector = new TopKnnCollector(10, Integer.MAX_VALUE);
        if (vectorsReaderSupportsSearch(codecReader, fieldInfo.name)) {
          codecReader
              .getVectorReader()
              .search(fieldInfo.name, values.vectorValue(count), collector, null);
          TopDocs docs = collector.topDocs();
          if (docs.scoreDocs.length == 0) {
            throw new CheckIndexException(
                "Field \"" + fieldInfo.name + "\" failed to search k nearest neighbors");
          }
        }
      }
      int valueLength = values.vectorValue(count).length;
      if (valueLength != fieldInfo.getVectorDimension()) {
        throw new CheckIndexException(
            "Field \""
                + fieldInfo.name
                + "\" has a value whose dimension="
                + valueLength
                + " not matching the field's dimension="
                + fieldInfo.getVectorDimension());
      }
      ++count;
    }
    if (count != values.size()) {
      throw new CheckIndexException(
          "Field \""
              + fieldInfo.name
              + "\" has size="
              + values.size()
              + " but when iterated, returns "
              + count
              + " docs with values");
    }
    status.totalVectorValues += count;
  }

  private static void checkByteVectorValues(
      ByteVectorValues values,
      FieldInfo fieldInfo,
      CheckIndex.Status.VectorValuesStatus status,
      CodecReader codecReader)
      throws IOException {
    int count = 0;
    int everyNdoc = Math.max(values.size() / 64, 1);
    boolean supportsSearch = vectorsReaderSupportsSearch(codecReader, fieldInfo.name);
    while (count < values.size()) {
      // search the first maxNumSearches vectors to exercise the graph
      if (supportsSearch && values.ordToDoc(count) % everyNdoc == 0) {
        KnnCollector collector = new TopKnnCollector(10, Integer.MAX_VALUE);
        codecReader
            .getVectorReader()
            .search(fieldInfo.name, values.vectorValue(count), collector, null);
        TopDocs docs = collector.topDocs();
        if (docs.scoreDocs.length == 0) {
          throw new CheckIndexException(
              "Field \"" + fieldInfo.name + "\" failed to search k nearest neighbors");
        }
      }
      int valueLength = values.vectorValue(count).length;
      if (valueLength != fieldInfo.getVectorDimension()) {
        throw new CheckIndexException(
            "Field \""
                + fieldInfo.name
                + "\" has a value whose dimension="
                + valueLength
                + " not matching the field's dimension="
                + fieldInfo.getVectorDimension());
      }
      ++count;
    }
    if (count != values.size()) {
      throw new CheckIndexException(
          "Field \""
              + fieldInfo.name
              + "\" has size="
              + values.size()
              + " but when iterated, returns "
              + count
              + " docs with values");
    }
    status.totalVectorValues += count;
  }

  /**
   * Walks the entire N-dimensional points space, verifying that all points fall within the last
   * cell's boundaries.
   *
   * @lucene.internal
   */
  public static class VerifyPointsVisitor implements PointValues.IntersectVisitor {
    private long pointCountSeen;
    private int lastDocID = -1;
    private final FixedBitSet docsSeen;
    private final byte[] lastMinPackedValue;
    private final byte[] lastMaxPackedValue;
    private final byte[] lastPackedValue;
    private final byte[] globalMinPackedValue;
    private final byte[] globalMaxPackedValue;
    private final int packedBytesCount;
    private final int packedIndexBytesCount;
    private final int numDataDims;
    private final int numIndexDims;
    private final int bytesPerDim;
    private final ByteArrayComparator comparator;
    private final String fieldName;

    /** Sole constructor */
    public VerifyPointsVisitor(String fieldName, int maxDoc, PointValues values)
        throws IOException {
      this.fieldName = fieldName;
      numDataDims = values.getNumDimensions();
      numIndexDims = values.getNumIndexDimensions();
      bytesPerDim = values.getBytesPerDimension();
      comparator = ArrayUtil.getUnsignedComparator(bytesPerDim);
      packedBytesCount = numDataDims * bytesPerDim;
      packedIndexBytesCount = numIndexDims * bytesPerDim;
      globalMinPackedValue = values.getMinPackedValue();
      globalMaxPackedValue = values.getMaxPackedValue();
      docsSeen = new FixedBitSet(maxDoc);
      lastMinPackedValue = new byte[packedIndexBytesCount];
      lastMaxPackedValue = new byte[packedIndexBytesCount];
      lastPackedValue = new byte[packedBytesCount];

      if (values.getDocCount() > values.size()) {
        throw new CheckIndexException(
            "point values for field \""
                + fieldName
                + "\" claims to have size="
                + values.size()
                + " points and inconsistent docCount="
                + values.getDocCount());
      }

      if (values.getDocCount() > maxDoc) {
        throw new CheckIndexException(
            "point values for field \""
                + fieldName
                + "\" claims to have docCount="
                + values.getDocCount()
                + " but that's greater than maxDoc="
                + maxDoc);
      }

      if (globalMinPackedValue == null) {
        if (values.size() != 0) {
          throw new CheckIndexException(
              "getMinPackedValue is null points for field \""
                  + fieldName
                  + "\" yet size="
                  + values.size());
        }
      } else if (globalMinPackedValue.length != packedIndexBytesCount) {
        throw new CheckIndexException(
            "getMinPackedValue for field \""
                + fieldName
                + "\" return length="
                + globalMinPackedValue.length
                + " array, but should be "
                + packedBytesCount);
      }
      if (globalMaxPackedValue == null) {
        if (values.size() != 0) {
          throw new CheckIndexException(
              "getMaxPackedValue is null points for field \""
                  + fieldName
                  + "\" yet size="
                  + values.size());
        }
      } else if (globalMaxPackedValue.length != packedIndexBytesCount) {
        throw new CheckIndexException(
            "getMaxPackedValue for field \""
                + fieldName
                + "\" return length="
                + globalMaxPackedValue.length
                + " array, but should be "
                + packedBytesCount);
      }
    }

    /** Returns total number of points in this BKD tree */
    public long getPointCountSeen() {
      return pointCountSeen;
    }

    /** Returns total number of unique docIDs in this BKD tree */
    public long getDocCountSeen() {
      return docsSeen.cardinality();
    }

    @Override
    public void visit(int docID) {
      throw new CheckIndexException(
          "codec called IntersectVisitor.visit without a packed value for docID=" + docID);
    }

    @Override
    public void visit(int docID, byte[] packedValue) {
      checkPackedValue("packed value", packedValue, docID);
      pointCountSeen++;
      docsSeen.set(docID);

      for (int dim = 0; dim < numIndexDims; dim++) {
        int offset = bytesPerDim * dim;

        // Compare to last cell:
        if (comparator.compare(packedValue, offset, lastMinPackedValue, offset) < 0) {
          // This doc's point, in this dimension, is lower than the minimum value of the last cell
          // checked:
          throw new CheckIndexException(
              "packed points value "
                  + Arrays.toString(packedValue)
                  + " for field=\""
                  + fieldName
                  + "\", docID="
                  + docID
                  + " is out-of-bounds of the last cell min="
                  + Arrays.toString(lastMinPackedValue)
                  + " max="
                  + Arrays.toString(lastMaxPackedValue)
                  + " dim="
                  + dim);
        }

        if (comparator.compare(packedValue, offset, lastMaxPackedValue, offset) > 0) {
          // This doc's point, in this dimension, is greater than the maximum value of the last cell
          // checked:
          throw new CheckIndexException(
              "packed points value "
                  + Arrays.toString(packedValue)
                  + " for field=\""
                  + fieldName
                  + "\", docID="
                  + docID
                  + " is out-of-bounds of the last cell min="
                  + Arrays.toString(lastMinPackedValue)
                  + " max="
                  + Arrays.toString(lastMaxPackedValue)
                  + " dim="
                  + dim);
        }
      }

      // In the 1D data case, PointValues must make a single in-order sweep through all values, and
      // tie-break by
      // increasing docID:
      // for data dimension > 1, leaves are sorted by the dimension with the lowest cardinality to
      // improve block compression
      if (numDataDims == 1) {
        int cmp = comparator.compare(lastPackedValue, 0, packedValue, 0);
        if (cmp > 0) {
          throw new CheckIndexException(
              "packed points value "
                  + Arrays.toString(packedValue)
                  + " for field=\""
                  + fieldName
                  + "\", for docID="
                  + docID
                  + " is out-of-order vs the previous document's value "
                  + Arrays.toString(lastPackedValue));
        } else if (cmp == 0) {
          if (docID < lastDocID) {
            throw new CheckIndexException(
                "packed points value is the same, but docID="
                    + docID
                    + " is out of order vs previous docID="
                    + lastDocID
                    + ", field=\""
                    + fieldName
                    + "\"");
          }
        }
        System.arraycopy(packedValue, 0, lastPackedValue, 0, bytesPerDim);
        lastDocID = docID;
      }
    }

    @Override
    public PointValues.Relation compare(byte[] minPackedValue, byte[] maxPackedValue) {
      checkPackedValue("min packed value", minPackedValue, -1);
      System.arraycopy(minPackedValue, 0, lastMinPackedValue, 0, packedIndexBytesCount);
      checkPackedValue("max packed value", maxPackedValue, -1);
      System.arraycopy(maxPackedValue, 0, lastMaxPackedValue, 0, packedIndexBytesCount);

      for (int dim = 0; dim < numIndexDims; dim++) {
        int offset = bytesPerDim * dim;

        if (comparator.compare(minPackedValue, offset, maxPackedValue, offset) > 0) {
          throw new CheckIndexException(
              "packed points cell minPackedValue "
                  + Arrays.toString(minPackedValue)
                  + " is out-of-bounds of the cell's maxPackedValue "
                  + Arrays.toString(maxPackedValue)
                  + " dim="
                  + dim
                  + " field=\""
                  + fieldName
                  + "\"");
        }

        // Make sure this cell is not outside the global min/max:
        if (comparator.compare(minPackedValue, offset, globalMinPackedValue, offset) < 0) {
          throw new CheckIndexException(
              "packed points cell minPackedValue "
                  + Arrays.toString(minPackedValue)
                  + " is out-of-bounds of the global minimum "
                  + Arrays.toString(globalMinPackedValue)
                  + " dim="
                  + dim
                  + " field=\""
                  + fieldName
                  + "\"");
        }

        if (comparator.compare(maxPackedValue, offset, globalMinPackedValue, offset) < 0) {
          throw new CheckIndexException(
              "packed points cell maxPackedValue "
                  + Arrays.toString(maxPackedValue)
                  + " is out-of-bounds of the global minimum "
                  + Arrays.toString(globalMinPackedValue)
                  + " dim="
                  + dim
                  + " field=\""
                  + fieldName
                  + "\"");
        }

        if (comparator.compare(minPackedValue, offset, globalMaxPackedValue, offset) > 0) {
          throw new CheckIndexException(
              "packed points cell minPackedValue "
                  + Arrays.toString(minPackedValue)
                  + " is out-of-bounds of the global maximum "
                  + Arrays.toString(globalMaxPackedValue)
                  + " dim="
                  + dim
                  + " field=\""
                  + fieldName
                  + "\"");
        }
        if (comparator.compare(maxPackedValue, offset, globalMaxPackedValue, offset) > 0) {
          throw new CheckIndexException(
              "packed points cell maxPackedValue "
                  + Arrays.toString(maxPackedValue)
                  + " is out-of-bounds of the global maximum "
                  + Arrays.toString(globalMaxPackedValue)
                  + " dim="
                  + dim
                  + " field=\""
                  + fieldName
                  + "\"");
        }
      }

      // We always pretend the query shape is so complex that it crosses every cell, so
      // that packedValue is passed for every document
      return PointValues.Relation.CELL_CROSSES_QUERY;
    }

    private void checkPackedValue(String desc, byte[] packedValue, int docID) {
      if (packedValue == null) {
        throw new CheckIndexException(
            desc + " is null for docID=" + docID + " field=\"" + fieldName + "\"");
      }

      if (packedValue.length != (docID < 0 ? packedIndexBytesCount : packedBytesCount)) {
        throw new CheckIndexException(
            desc
                + " has incorrect length="
                + packedValue.length
                + " vs expected="
                + packedIndexBytesCount
                + " for docID="
                + docID
                + " field=\""
                + fieldName
                + "\"");
      }
    }
  }

  private record ConstantRelationIntersectVisitor(Relation relation) implements IntersectVisitor {

    @Override
    public void visit(int docID) throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public void visit(int docID, byte[] packedValue) throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public Relation compare(byte[] minPackedValue, byte[] maxPackedValue) {
      return relation;
    }
  }

  /** Test stored fields. */
  public static Status.StoredFieldStatus testStoredFields(
      CodecReader reader, PrintStream infoStream, boolean failFast) throws IOException {
    long startNS = System.nanoTime();
    final Status.StoredFieldStatus status = new Status.StoredFieldStatus();

    try {
      if (infoStream != null) {
        infoStream.print("    test: stored fields.......");
      }

      // Scan stored fields for all documents
      final Bits liveDocs = reader.getLiveDocs();
      StoredFieldsReader storedFields = reader.getFieldsReader().getMergeInstance();
      for (int j = 0; j < reader.maxDoc(); ++j) {
        // Intentionally pull even deleted documents to
        // make sure they too are not corrupt:
        DocumentStoredFieldVisitor visitor = new DocumentStoredFieldVisitor();
        if ((j & 0x03) == 0) {
          storedFields.prefetch(j);
        }
        storedFields.document(j, visitor);
        Document doc = visitor.getDocument();
        if (liveDocs == null || liveDocs.get(j)) {
          status.docCount++;
          status.totFields += doc.getFields().size();
        }
      }

      // Validate docCount
      if (status.docCount != reader.numDocs()) {
        throw new CheckIndexException(
            "docCount=" + status.docCount + " but saw " + status.docCount + " undeleted docs");
      }

      msg(
          infoStream,
          String.format(
              Locale.ROOT,
              "OK [%d total field count; avg %.1f fields per doc] [took %.3f sec]",
              status.totFields,
              (((float) status.totFields) / status.docCount),
              nsToSec(System.nanoTime() - startNS)));
    } catch (Throwable e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR [" + e.getMessage() + "]");
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }

    return status;
  }

  /** Test docvalues. */
  public static Status.DocValuesStatus testDocValues(
      CodecReader reader, PrintStream infoStream, boolean failFast) throws IOException {
    long startNS = System.nanoTime();

    final Status.DocValuesStatus status = new Status.DocValuesStatus();
    try {
      if (infoStream != null) {
        infoStream.print("    test: docvalues...........");
      }
      DocValuesProducer dvReader = reader.getDocValuesReader();
      if (dvReader != null) {
        dvReader = dvReader.getMergeInstance();
      }
      for (FieldInfo fieldInfo : reader.getFieldInfos()) {
        if (fieldInfo.getDocValuesType() != DocValuesType.NONE) {
          status.totalValueFields++;
          checkDocValues(fieldInfo, dvReader, status);
        }
      }

      msg(
          infoStream,
          String.format(
              Locale.ROOT,
              "OK [%d docvalues fields; %d BINARY; %d NUMERIC; %d SORTED; %d SORTED_NUMERIC; %d SORTED_SET; %d SKIPPING INDEX] [took %.3f sec]",
              status.totalValueFields,
              status.totalBinaryFields,
              status.totalNumericFields,
              status.totalSortedFields,
              status.totalSortedNumericFields,
              status.totalSortedSetFields,
              status.totalSkippingIndex,
              nsToSec(System.nanoTime() - startNS)));
    } catch (Throwable e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR [" + e.getMessage() + "]");
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }
    return status;
  }

  private static void checkDocValueSkipper(FieldInfo fi, DocValuesSkipper skipper)
      throws IOException {
    String fieldName = fi.name;
    if (skipper.maxDocID(0) != -1) {
      throw new CheckIndexException(
          "binary dv iterator for field: "
              + fieldName
              + " should start at docID=-1, but got "
              + skipper.maxDocID(0));
    }
    if (skipper.docCount() > 0 && skipper.minValue() > skipper.maxValue()) {
      throw new CheckIndexException(
          "skipper dv iterator for field: "
              + fieldName
              + " reports wrong global value range, got  "
              + skipper.minValue()
              + " > "
              + skipper.maxValue());
    }
    int docCount = 0;
    int doc;
    while (true) {
      doc = skipper.maxDocID(0) + 1;
      skipper.advance(doc);
      if (skipper.maxDocID(0) == NO_MORE_DOCS) {
        break;
      }
      if (skipper.minDocID(0) < doc) {
        throw new CheckIndexException(
            "skipper dv iterator for field: "
                + fieldName
                + " reports wrong minDocID, got "
                + skipper.minDocID(0)
                + " < "
                + doc);
      }
      int levels = skipper.numLevels();
      for (int level = 0; level < levels; level++) {
        if (skipper.minDocID(level) > skipper.maxDocID(level)) {
          throw new CheckIndexException(
              "skipper dv iterator for field: "
                  + fieldName
                  + " reports wrong doc range, got "
                  + skipper.minDocID(level)
                  + " > "
                  + skipper.maxDocID(level));
        }
        if (skipper.minValue() > skipper.minValue(level)) {
          throw new CheckIndexException(
              "skipper dv iterator for field: "
                  + fieldName
                  + " : global minValue  "
                  + skipper.minValue()
                  + " , got  "
                  + skipper.minValue(level));
        }
        if (skipper.maxValue() < skipper.maxValue(level)) {
          throw new CheckIndexException(
              "skipper dv iterator for field: "
                  + fieldName
                  + " : global maxValue  "
                  + skipper.maxValue()
                  + " , got  "
                  + skipper.maxValue(level));
        }
        if (skipper.minValue(level) > skipper.maxValue(level)) {
          throw new CheckIndexException(
              "skipper dv iterator for field: "
                  + fieldName
                  + " reports wrong value range, got  "
                  + skipper.minValue(level)
                  + " > "
                  + skipper.maxValue(level));
        }
      }
      docCount += skipper.docCount(0);
    }
    if (skipper.docCount() != docCount) {
      throw new CheckIndexException(
          "skipper dv iterator for field: "
              + fieldName
              + " inconsistent docCount, got "
              + skipper.docCount()
              + " != "
              + docCount);
    }
  }

  private static void checkDVIterator(
      FieldInfo fi, IOFunction<FieldInfo, DocValuesIterator> producer) throws IOException {
    String field = fi.name;

    // Check advance
    DocValuesIterator it1 = producer.apply(fi);
    DocValuesIterator it2 = producer.apply(fi);
    int i = 0;
    for (int doc = it1.nextDoc(); ; doc = it1.nextDoc()) {

      if (i++ % 10 == 1) {
        int doc2 = it2.advance(doc - 1);
        if (doc2 < doc - 1) {
          throw new CheckIndexException(
              "dv iterator field="
                  + field
                  + ": doc="
                  + (doc - 1)
                  + " went backwords (got: "
                  + doc2
                  + ")");
        }
        if (doc2 == doc - 1) {
          doc2 = it2.nextDoc();
        }
        if (doc2 != doc) {
          throw new CheckIndexException(
              "dv iterator field="
                  + field
                  + ": doc="
                  + doc
                  + " was not found through advance() (got: "
                  + doc2
                  + ")");
        }
        if (it2.docID() != doc) {
          throw new CheckIndexException(
              "dv iterator field="
                  + field
                  + ": doc="
                  + doc
                  + " reports wrong doc ID (got: "
                  + it2.docID()
                  + ")");
        }
      }

      if (doc == NO_MORE_DOCS) {
        break;
      }
    }

    // Check advanceExact
    it1 = producer.apply(fi);
    it2 = producer.apply(fi);
    i = 0;
    int lastDoc = -1;
    for (int doc = it1.nextDoc(); doc != NO_MORE_DOCS; doc = it1.nextDoc()) {

      if (i++ % 13 == 1) {
        boolean found = it2.advanceExact(doc - 1);
        if ((doc - 1 == lastDoc) != found) {
          throw new CheckIndexException(
              "dv iterator field="
                  + field
                  + ": doc="
                  + (doc - 1)
                  + " disagrees about whether document exists (got: "
                  + found
                  + ")");
        }
        if (it2.docID() != doc - 1) {
          throw new CheckIndexException(
              "dv iterator field="
                  + field
                  + ": doc="
                  + (doc - 1)
                  + " reports wrong doc ID (got: "
                  + it2.docID()
                  + ")");
        }

        boolean found2 = it2.advanceExact(doc - 1);
        if (found != found2) {
          throw new CheckIndexException(
              "dv iterator field=" + field + ": doc=" + (doc - 1) + " has unstable advanceExact");
        }

        if (i % 2 == 0) {
          int doc2 = it2.nextDoc();
          if (doc != doc2) {
            throw new CheckIndexException(
                "dv iterator field="
                    + field
                    + ": doc="
                    + doc
                    + " was not found through advance() (got: "
                    + doc2
                    + ")");
          }
          if (it2.docID() != doc) {
            throw new CheckIndexException(
                "dv iterator field="
                    + field
                    + ": doc="
                    + doc
                    + " reports wrong doc ID (got: "
                    + it2.docID()
                    + ")");
          }
        }
      }

      lastDoc = doc;
    }
  }

  private static void checkBinaryDocValues(
      String fieldName, BinaryDocValues bdv, BinaryDocValues bdv2) throws IOException {
    if (bdv.docID() != -1) {
      throw new CheckIndexException(
          "binary dv iterator for field: "
              + fieldName
              + " should start at docID=-1, but got "
              + bdv.docID());
    }
    // TODO: we could add stats to DVs, e.g. total doc count w/ a value for this field
    for (int doc = bdv.nextDoc(); doc != NO_MORE_DOCS; doc = bdv.nextDoc()) {
      BytesRef value = bdv.binaryValue();
      value.isValid();

      if (bdv2.advanceExact(doc) == false) {
        throw new CheckIndexException("advanceExact did not find matching doc ID: " + doc);
      }
      BytesRef value2 = bdv2.binaryValue();
      if (value.equals(value2) == false) {
        throw new CheckIndexException(
            "nextDoc and advanceExact report different values: " + value + " != " + value2);
      }
    }
  }

  private static void checkSortedDocValues(
      String fieldName, SortedDocValues dv, SortedDocValues dv2) throws IOException {
    if (dv.docID() != -1) {
      throw new CheckIndexException(
          "sorted dv iterator for field: "
              + fieldName
              + " should start at docID=-1, but got "
              + dv.docID());
    }
    final int maxOrd = dv.getValueCount() - 1;
    FixedBitSet seenOrds = new FixedBitSet(dv.getValueCount());
    int maxOrd2 = -1;
    for (int doc = dv.nextDoc(); doc != NO_MORE_DOCS; doc = dv.nextDoc()) {
      int ord = dv.ordValue();
      if (ord == -1) {
        throw new CheckIndexException("dv for field: " + fieldName + " has -1 ord");
      } else if (ord < -1 || ord > maxOrd) {
        throw new CheckIndexException("ord out of bounds: " + ord);
      } else {
        maxOrd2 = Math.max(maxOrd2, ord);
        seenOrds.set(ord);
      }

      if (dv2.advanceExact(doc) == false) {
        throw new CheckIndexException("advanceExact did not find matching doc ID: " + doc);
      }
      int ord2 = dv2.ordValue();
      if (ord != ord2) {
        throw new CheckIndexException(
            "nextDoc and advanceExact report different ords: " + ord + " != " + ord2);
      }
    }
    if (maxOrd != maxOrd2) {
      throw new CheckIndexException(
          "dv for field: "
              + fieldName
              + " reports wrong maxOrd="
              + maxOrd
              + " but this is not the case: "
              + maxOrd2);
    }
    if (seenOrds.cardinality() != dv.getValueCount()) {
      throw new CheckIndexException(
          "dv for field: "
              + fieldName
              + " has holes in its ords, valueCount="
              + dv.getValueCount()
              + " but only used: "
              + seenOrds.cardinality());
    }
    BytesRef lastValue = null;
    for (int i = 0; i <= maxOrd; i++) {
      final BytesRef term = dv.lookupOrd(i);
      term.isValid();
      if (lastValue != null) {
        if (term.compareTo(lastValue) <= 0) {
          throw new CheckIndexException(
              "dv for field: " + fieldName + " has ords out of order: " + lastValue + " >=" + term);
        }
      }
      lastValue = BytesRef.deepCopyOf(term);
    }
  }

  private static void checkSortedSetDocValues(
      String fieldName, SortedSetDocValues dv, SortedSetDocValues dv2) throws IOException {
    final long maxOrd = dv.getValueCount() - 1;
    LongBitSet seenOrds = new LongBitSet(dv.getValueCount());
    long maxOrd2 = -1;
    for (int docID = dv.nextDoc(); docID != NO_MORE_DOCS; docID = dv.nextDoc()) {
      int count = dv.docValueCount();
      if (count == 0) {
        throw new CheckIndexException(
            "sortedset dv for field: "
                + fieldName
                + " returned docValueCount=0 for docID="
                + docID);
      }
      if (dv2.advanceExact(docID) == false) {
        throw new CheckIndexException("advanceExact did not find matching doc ID: " + docID);
      }
      int count2 = dv2.docValueCount();
      if (count != count2) {
        throw new CheckIndexException(
            "advanceExact reports different value count: " + count + " != " + count2);
      }
      long lastOrd = -1;
      int ordCount = 0;
      for (int i = 0; i < count; i++) {
        if (count != dv.docValueCount()) {
          throw new CheckIndexException(
              "value count changed from "
                  + count
                  + " to "
                  + dv.docValueCount()
                  + " during iterating over all values");
        }
        long ord = dv.nextOrd();
        long ord2 = dv2.nextOrd();
        if (ord != ord2) {
          throw new CheckIndexException(
              "nextDoc and advanceExact report different ords: " + ord + " != " + ord2);
        }
        if (ord <= lastOrd) {
          throw new CheckIndexException(
              "ords out of order: " + ord + " <= " + lastOrd + " for doc: " + docID);
        }
        if (ord < 0 || ord > maxOrd) {
          throw new CheckIndexException("ord out of bounds: " + ord);
        }
        lastOrd = ord;
        maxOrd2 = Math.max(maxOrd2, ord);
        seenOrds.set(ord);
        ordCount++;
      }
      if (dv.docValueCount() != dv2.docValueCount()) {
        throw new CheckIndexException(
            "dv and dv2 report different values count after iterating over all values: "
                + dv.docValueCount()
                + " != "
                + dv2.docValueCount());
      }
      if (ordCount == 0) {
        throw new CheckIndexException(
            "dv for field: " + fieldName + " returned docID=" + docID + " yet has no ordinals");
      }
    }
    if (maxOrd != maxOrd2) {
      throw new CheckIndexException(
          "dv for field: "
              + fieldName
              + " reports wrong maxOrd="
              + maxOrd
              + " but this is not the case: "
              + maxOrd2);
    }
    if (seenOrds.cardinality() != dv.getValueCount()) {
      throw new CheckIndexException(
          "dv for field: "
              + fieldName
              + " has holes in its ords, valueCount="
              + dv.getValueCount()
              + " but only used: "
              + seenOrds.cardinality());
    }

    BytesRef lastValue = null;
    for (long i = 0; i <= maxOrd; i++) {
      final BytesRef term = dv.lookupOrd(i);
      assert term.isValid();
      if (lastValue != null) {
        if (term.compareTo(lastValue) <= 0) {
          throw new CheckIndexException(
              "dv for field: " + fieldName + " has ords out of order: " + lastValue + " >=" + term);
        }
      }
      lastValue = BytesRef.deepCopyOf(term);
    }
  }

  private static void checkSortedNumericDocValues(
      String fieldName, SortedNumericDocValues ndv, SortedNumericDocValues ndv2)
      throws IOException {
    if (ndv.docID() != -1) {
      throw new CheckIndexException(
          "dv iterator for field: "
              + fieldName
              + " should start at docID=-1, but got "
              + ndv.docID());
    }
    for (int docID = ndv.nextDoc(); docID != NO_MORE_DOCS; docID = ndv.nextDoc()) {
      int count = ndv.docValueCount();
      if (count == 0) {
        throw new CheckIndexException(
            "sorted numeric dv for field: "
                + fieldName
                + " returned docValueCount=0 for docID="
                + docID);
      }
      if (ndv2.advanceExact(docID) == false) {
        throw new CheckIndexException("advanceExact did not find matching doc ID: " + docID);
      }
      int count2 = ndv2.docValueCount();
      if (count != count2) {
        throw new CheckIndexException(
            "advanceExact reports different value count: " + count + " != " + count2);
      }
      long previous = Long.MIN_VALUE;
      for (int j = 0; j < count; j++) {
        long value = ndv.nextValue();
        if (value < previous) {
          throw new CheckIndexException(
              "values out of order: " + value + " < " + previous + " for doc: " + docID);
        }
        previous = value;

        long value2 = ndv2.nextValue();
        if (value != value2) {
          throw new CheckIndexException(
              "advanceExact reports different value: " + value + " != " + value2);
        }
      }
    }
  }

  private static void checkNumericDocValues(
      String fieldName, NumericDocValues ndv, NumericDocValues ndv2) throws IOException {
    if (ndv.docID() != -1) {
      throw new CheckIndexException(
          "dv iterator for field: "
              + fieldName
              + " should start at docID=-1, but got "
              + ndv.docID());
    }
    // TODO: we could add stats to DVs, e.g. total doc count w/ a value for this field
    for (int doc = ndv.nextDoc(); doc != NO_MORE_DOCS; doc = ndv.nextDoc()) {
      long value = ndv.longValue();

      if (ndv2.advanceExact(doc) == false) {
        throw new CheckIndexException("advanceExact did not find matching doc ID: " + doc);
      }
      long value2 = ndv2.longValue();
      if (value != value2) {
        throw new CheckIndexException(
            "advanceExact reports different value: " + value + " != " + value2);
      }
    }
  }

  private static void checkDocValues(
      FieldInfo fi, DocValuesProducer dvReader, DocValuesStatus status) throws Exception {
    if (fi.docValuesSkipIndexType() != DocValuesSkipIndexType.NONE) {
      status.totalSkippingIndex++;
      checkDocValueSkipper(fi, dvReader.getSkipper(fi));
    }
    switch (fi.getDocValuesType()) {
      case SORTED:
        status.totalSortedFields++;
        checkDVIterator(fi, dvReader::getSorted);
        checkSortedDocValues(fi.name, dvReader.getSorted(fi), dvReader.getSorted(fi));
        break;
      case SORTED_NUMERIC:
        status.totalSortedNumericFields++;
        checkDVIterator(fi, dvReader::getSortedNumeric);
        checkSortedNumericDocValues(
            fi.name, dvReader.getSortedNumeric(fi), dvReader.getSortedNumeric(fi));
        break;
      case SORTED_SET:
        status.totalSortedSetFields++;
        checkDVIterator(fi, dvReader::getSortedSet);
        checkSortedSetDocValues(fi.name, dvReader.getSortedSet(fi), dvReader.getSortedSet(fi));
        break;
      case BINARY:
        status.totalBinaryFields++;
        checkDVIterator(fi, dvReader::getBinary);
        checkBinaryDocValues(fi.name, dvReader.getBinary(fi), dvReader.getBinary(fi));
        break;
      case NUMERIC:
        status.totalNumericFields++;
        checkDVIterator(fi, dvReader::getNumeric);
        checkNumericDocValues(fi.name, dvReader.getNumeric(fi), dvReader.getNumeric(fi));
        break;
      case NONE:
      default:
        throw new AssertionError();
    }
  }

  /** Test term vectors. */
  public static Status.TermVectorStatus testTermVectors(CodecReader reader, PrintStream infoStream)
      throws IOException {
    return testTermVectors(reader, infoStream, false, Level.MIN_LEVEL_FOR_INTEGRITY_CHECKS, false);
  }

  /** Test term vectors. */
  public static Status.TermVectorStatus testTermVectors(
      CodecReader reader, PrintStream infoStream, boolean verbose, int level, boolean failFast)
      throws IOException {
    long startNS = System.nanoTime();
    final Status.TermVectorStatus status = new Status.TermVectorStatus();
    final FieldInfos fieldInfos = reader.getFieldInfos();

    try {
      if (infoStream != null) {
        infoStream.print("    test: term vectors........");
      }

      PostingsEnum postings = null;

      // Only used if the Level is high enough to include slow checks:
      PostingsEnum postingsDocs = null;

      final Bits liveDocs = reader.getLiveDocs();

      FieldsProducer postingsFields;
      // TODO: testTermsIndex
      if (level >= Level.MIN_LEVEL_FOR_SLOW_CHECKS) {
        postingsFields = reader.getPostingsReader();
        if (postingsFields != null) {
          postingsFields = postingsFields.getMergeInstance();
        }
      } else {
        postingsFields = null;
      }

      TermVectorsReader vectorsReader = reader.getTermVectorsReader();

      if (vectorsReader != null) {
        vectorsReader = vectorsReader.getMergeInstance();
        for (int j = 0; j < reader.maxDoc(); ++j) {
          if ((j & 0x03) == 0) {
            vectorsReader.prefetch(j);
          }
          // Intentionally pull/visit (but don't count in
          // stats) deleted documents to make sure they too
          // are not corrupt:
          Fields tfv = vectorsReader.get(j);

          // TODO: can we make a IS(FIR) that searches just
          // this term vector... to pass for searcher?

          if (tfv != null) {
            // First run with no deletions:
            checkFields(tfv, null, 1, fieldInfos, null, false, true, infoStream, verbose, level);

            // Only agg stats if the doc is live:
            final boolean doStats = liveDocs == null || liveDocs.get(j);

            if (doStats) {
              status.docCount++;
            }

            for (String field : tfv) {
              if (doStats) {
                status.totVectors++;
              }

              // Make sure FieldInfo thinks this field is vector'd:
              final FieldInfo fieldInfo = fieldInfos.fieldInfo(field);
              if (fieldInfo.hasTermVectors() == false) {
                throw new CheckIndexException(
                    "docID="
                        + j
                        + " has term vectors for field="
                        + field
                        + " but FieldInfo has storeTermVector=false");
              }

              if (level >= Level.MIN_LEVEL_FOR_SLOW_CHECKS) {
                Terms terms = tfv.terms(field);
                TermsEnum termsEnum = terms.iterator();
                final boolean postingsHasFreq =
                    fieldInfo.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS) >= 0;
                final boolean postingsHasPayload = fieldInfo.hasPayloads();
                final boolean vectorsHasPayload = terms.hasPayloads();

                if (postingsFields == null) {
                  throw new CheckIndexException(
                      "vector field=" + field + " does not exist in postings; doc=" + j);
                }
                Terms postingsTerms = postingsFields.terms(field);
                if (postingsTerms == null) {
                  throw new CheckIndexException(
                      "vector field=" + field + " does not exist in postings; doc=" + j);
                }
                TermsEnum postingsTermsEnum = postingsTerms.iterator();

                final boolean hasProx = terms.hasOffsets() || terms.hasPositions();
                int seekExactCounter = 0;
                BytesRef term;
                while ((term = termsEnum.next()) != null) {

                  // This is the term vectors:
                  postings = termsEnum.postings(postings, PostingsEnum.ALL);
                  assert postings != null;

                  boolean termExists;
                  if ((seekExactCounter++ & 0x01) == 0) {
                    termExists = postingsTermsEnum.seekExact(term);
                  } else {
                    IOBooleanSupplier termExistsSupplier = postingsTermsEnum.prepareSeekExact(term);
                    termExists = termExistsSupplier != null && termExistsSupplier.get();
                  }
                  if (termExists == false) {
                    throw new CheckIndexException(
                        "vector term="
                            + term
                            + " field="
                            + field
                            + " does not exist in postings; doc="
                            + j);
                  }

                  // This is the inverted index ("real" postings):
                  postingsDocs = postingsTermsEnum.postings(postingsDocs, PostingsEnum.ALL);
                  assert postingsDocs != null;

                  final int advanceDoc = postingsDocs.advance(j);
                  if (advanceDoc != j) {
                    throw new CheckIndexException(
                        "vector term="
                            + term
                            + " field="
                            + field
                            + ": doc="
                            + j
                            + " was not found in postings (got: "
                            + advanceDoc
                            + ")");
                  }

                  final int doc = postings.nextDoc();

                  if (doc != 0) {
                    throw new CheckIndexException(
                        "vector for doc " + j + " didn't return docID=0: got docID=" + doc);
                  }

                  if (postingsHasFreq) {
                    final int tf = postings.freq();
                    if (postingsHasFreq && postingsDocs.freq() != tf) {
                      throw new CheckIndexException(
                          "vector term="
                              + term
                              + " field="
                              + field
                              + " doc="
                              + j
                              + ": freq="
                              + tf
                              + " differs from postings freq="
                              + postingsDocs.freq());
                    }

                    // Term vectors has prox?
                    if (hasProx) {
                      for (int i = 0; i < tf; i++) {
                        int pos = postings.nextPosition();
                        if (postingsTerms.hasPositions()) {
                          int postingsPos = postingsDocs.nextPosition();
                          if (terms.hasPositions() && pos != postingsPos) {
                            throw new CheckIndexException(
                                "vector term="
                                    + term
                                    + " field="
                                    + field
                                    + " doc="
                                    + j
                                    + ": pos="
                                    + pos
                                    + " differs from postings pos="
                                    + postingsPos);
                          }
                        }

                        // Call the methods to at least make
                        // sure they don't throw exc:
                        final int startOffset = postings.startOffset();
                        final int endOffset = postings.endOffset();
                        // TODO: these are too anal...?
                        /*
                        if (endOffset < startOffset) {
                        throw new RuntimeException("vector startOffset=" + startOffset + " is > endOffset=" + endOffset);
                        }
                        if (startOffset < lastStartOffset) {
                        throw new RuntimeException("vector startOffset=" + startOffset + " is < prior startOffset=" + lastStartOffset);
                        }
                        lastStartOffset = startOffset;
                         */

                        if (startOffset != -1 && endOffset != -1 && postingsTerms.hasOffsets()) {
                          int postingsStartOffset = postingsDocs.startOffset();
                          int postingsEndOffset = postingsDocs.endOffset();
                          if (startOffset != postingsStartOffset) {
                            throw new CheckIndexException(
                                "vector term="
                                    + term
                                    + " field="
                                    + field
                                    + " doc="
                                    + j
                                    + ": startOffset="
                                    + startOffset
                                    + " differs from postings startOffset="
                                    + postingsStartOffset);
                          }
                          if (endOffset != postingsEndOffset) {
                            throw new CheckIndexException(
                                "vector term="
                                    + term
                                    + " field="
                                    + field
                                    + " doc="
                                    + j
                                    + ": endOffset="
                                    + endOffset
                                    + " differs from postings endOffset="
                                    + postingsEndOffset);
                          }
                        }

                        BytesRef payload = postings.getPayload();

                        if (payload != null) {
                          assert vectorsHasPayload;
                        }

                        if (postingsHasPayload && vectorsHasPayload) {

                          if (payload == null) {
                            // we have payloads, but not at this position.
                            // postings has payloads too, it should not have one at this position
                            if (postingsDocs.getPayload() != null) {
                              throw new CheckIndexException(
                                  "vector term="
                                      + term
                                      + " field="
                                      + field
                                      + " doc="
                                      + j
                                      + " has no payload but postings does: "
                                      + postingsDocs.getPayload());
                            }
                          } else {
                            // we have payloads, and one at this position
                            // postings should also have one at this position, with the same bytes.
                            if (postingsDocs.getPayload() == null) {
                              throw new CheckIndexException(
                                  "vector term="
                                      + term
                                      + " field="
                                      + field
                                      + " doc="
                                      + j
                                      + " has payload="
                                      + payload
                                      + " but postings does not.");
                            }
                            BytesRef postingsPayload = postingsDocs.getPayload();
                            if (payload.equals(postingsPayload) == false) {
                              throw new CheckIndexException(
                                  "vector term="
                                      + term
                                      + " field="
                                      + field
                                      + " doc="
                                      + j
                                      + " has payload="
                                      + payload
                                      + " but differs from postings payload="
                                      + postingsPayload);
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      float vectorAvg = status.docCount == 0 ? 0 : status.totVectors / (float) status.docCount;
      msg(
          infoStream,
          String.format(
              Locale.ROOT,
              "OK [%d total term vector count; avg %.1f term/freq vector fields per doc] [took %.3f sec]",
              status.totVectors,
              vectorAvg,
              nsToSec(System.nanoTime() - startNS)));
    } catch (Throwable e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR [" + e.getMessage() + "]");
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }

    return status;
  }

  /**
   * Repairs the index using previously returned result from {@link #checkIndex}. Note that this
   * does not remove any of the unreferenced files after it's done; you must separately open an
   * {@link IndexWriter}, which deletes unreferenced files when it's created.
   *
   * <p><b>WARNING</b>: this writes a new segments file into the index, effectively removing all
   * documents in broken segments from the index. BE CAREFUL.
   */
  public void exorciseIndex(Status result) throws IOException {
    ensureOpen();
    if (result.partial) {
      throw new IllegalArgumentException(
          "can only exorcise an index that was fully checked (this status checked a subset of segments)");
    }
    result.newSegments.changed();
    result.newSegments.commit(result.dir);
  }

  @SuppressWarnings("NonFinalStaticField")
  private static boolean assertsOn;

  private static boolean testAsserts() {
    assertsOn = true;
    return true;
  }

  /**
   * Check whether asserts are enabled or not.
   *
   * @return true iff asserts are enabled
   */
  public static boolean assertsOn() {
    assert testAsserts();
    return assertsOn;
  }

  /**
   * Command-line interface to check and exorcise corrupt segments from an index.
   *
   * <p>Run it like this:
   *
   * <pre>
   * java -ea:org.apache.lucene... org.apache.lucene.index.CheckIndex pathToIndex [-exorcise] [-verbose] [-segment X] [-segment Y]
   * </pre>
   *
   * <ul>
   *   <li><code>-exorcise</code>: actually write a new segments_N file, removing any problematic
   *       segments. *LOSES DATA*
   *   <li><code>-segment X</code>: only check the specified segment(s). This can be specified
   *       multiple times, to check more than one segment: <code>-segment _2 * -segment _a</code>.
   *       You can't use this with the -exorcise option.
   * </ul>
   *
   * <p><b>WARNING</b>: <code>-exorcise</code> should only be used on an emergency basis as it will
   * cause documents (perhaps many) to be permanently removed from the index. Always make a backup
   * copy of your index before running this! Do not run this tool on an index that is actively being
   * written to. You have been warned!
   *
   * <p>Run without -exorcise, this tool will open the index, report version information and report
   * any exceptions it hits and what action it would take if -exorcise were specified. With
   * -exorcise, this tool will remove any segments that have issues and write a new segments_N file.
   * This means all documents contained in the affected segments will be removed.
   *
   * <p>This tool exits with exit code 1 if the index cannot be opened or has any corruption, else
   * 0.
   */
  public static void main(String[] args) throws IOException, InterruptedException {
    int exitCode = doMain(args);
    System.exit(exitCode);
  }

  /** Run-time configuration options for CheckIndex commands. */
  public static class Options {
    boolean doExorcise = false;
    boolean verbose = false;
    int level = Level.DEFAULT_VALUE;
    int threadCount;
    List<String> onlySegments = new ArrayList<>();
    String indexPath = null;
    String dirImpl = null;
    PrintStream out = null;

    /** Sole constructor. */
    public Options() {}

    /** Get the name of the FSDirectory implementation class to use. */
    public String getDirImpl() {
      return dirImpl;
    }

    /** Get the directory containing the index. */
    public String getIndexPath() {
      return indexPath;
    }

    /** Set the PrintStream to use for reporting results. */
    public void setOut(PrintStream out) {
      this.out = out;
    }
  }

  // actual main: returns exit code instead of terminating JVM (for easy testing)
  @SuppressForbidden(reason = "System.out required: command line tool")
  private static int doMain(String[] args) throws IOException, InterruptedException {
    Options opts;
    try {
      opts = parseOptions(args);
    } catch (IllegalArgumentException e) {
      System.out.println(e.getMessage());
      return 1;
    }

    if (assertsOn() == false) {
      System.out.println(
          "\nNOTE: testing will be more thorough if you run java with '-ea:org.apache.lucene...', so assertions are enabled");
    }

    System.out.println("\nOpening index @ " + opts.indexPath + "\n");
    Directory directory;
    Path path = Paths.get(opts.indexPath);
    try {
      if (opts.dirImpl == null) {
        directory = FSDirectory.open(path);
      } else {
        directory = CommandLineUtil.newFSDirectory(opts.dirImpl, path);
      }
    } catch (Throwable t) {
      System.out.println("ERROR: could not open directory \"" + opts.indexPath + "\"; exiting");
      t.printStackTrace(System.out);
      return 1;
    }

    try (Directory dir = directory;
        CheckIndex checker = new CheckIndex(dir)) {
      opts.out = System.out;
      return checker.doCheck(opts);
    }
  }

  /** Class with static variables with information about CheckIndex's -level parameter. */
  public static class Level {
    private Level() {}

    /** Minimum valid level. */
    public static final int MIN_VALUE = 1;

    /** Maximum valid level. */
    public static final int MAX_VALUE = 3;

    /** The default level if none is specified. */
    public static final int DEFAULT_VALUE = MIN_VALUE;

    /** Minimum level required to run checksum checks. */
    public static final int MIN_LEVEL_FOR_CHECKSUM_CHECKS = 1;

    /** Minimum level required to run integrity checks. */
    public static final int MIN_LEVEL_FOR_INTEGRITY_CHECKS = 2;

    /** Minimum level required to run slow checks. */
    public static final int MIN_LEVEL_FOR_SLOW_CHECKS = 3;

    /** Checks if given level value is within the allowed bounds else it raises an Exception. */
    public static void checkIfLevelInBounds(int levelVal) throws IllegalArgumentException {
      if (levelVal < Level.MIN_VALUE || levelVal > Level.MAX_VALUE) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT,
                "ERROR: given value: '%d' for -level option is out of bounds. Please use a value from '%d'->'%d'",
                levelVal,
                Level.MIN_VALUE,
                Level.MAX_VALUE));
      }
    }
  }

  /**
   * Parse command line args into fields
   *
   * @param args The command line arguments
   * @return An Options struct
   * @throws IllegalArgumentException if any of the CLI args are invalid
   */
  @SuppressForbidden(reason = "System.err required: command line tool")
  public static Options parseOptions(String[] args) {
    Options opts = new Options();

    int i = 0;
    while (i < args.length) {
      String arg = args[i];
      if ("-level".equals(arg)) {
        if (i == args.length - 1) {
          throw new IllegalArgumentException("ERROR: missing value for -level option");
        }
        i++;
        int level = Integer.parseInt(args[i]);
        Level.checkIfLevelInBounds(level);
        opts.level = level;
      } else if ("-exorcise".equals(arg)) {
        opts.doExorcise = true;
      } else if (arg.equals("-verbose")) {
        opts.verbose = true;
      } else if (arg.equals("-segment")) {
        if (i == args.length - 1) {
          throw new IllegalArgumentException("ERROR: missing name for -segment option");
        }
        i++;
        opts.onlySegments.add(args[i]);
      } else if ("-dir-impl".equals(arg)) {
        if (i == args.length - 1) {
          throw new IllegalArgumentException("ERROR: missing value for -dir-impl option");
        }
        i++;
        opts.dirImpl = args[i];
      } else if ("-threadCount".equals(arg)) {
        if (i == args.length - 1) {
          throw new IllegalArgumentException("-threadCount requires a following number");
        }
        i++;
        opts.threadCount = Integer.parseInt(args[i]);
        if (opts.threadCount <= 0) {
          throw new IllegalArgumentException(
              "-threadCount requires a number larger than 0, but got: " + opts.threadCount);
        }
      } else {
        if (opts.indexPath != null) {
          throw new IllegalArgumentException("ERROR: unexpected extra argument '" + args[i] + "'");
        }
        opts.indexPath = args[i];
      }
      i++;
    }

    if (opts.indexPath == null) {
      throw new IllegalArgumentException(
          "\nERROR: index path not specified"
              + "\nUsage: java org.apache.lucene.index.CheckIndex pathToIndex [-exorcise] [-level X] [-segment X] [-segment Y] [-threadCount X] [-dir-impl X]\n"
              + "\n"
              + "  -exorcise: actually write a new segments_N file, removing any problematic segments\n"
              + "  -level X: sets the detail level of the check. The higher the value, the more checks are done.\n"
              + "         1 - (Default) Checksum checks only.\n"
              + "         2 - All level 1 checks + logical integrity checks.\n"
              + "         3 - All level 2 checks + slow checks.\n"
              + "  -codec X: when exorcising, codec to write the new segments_N file with\n"
              + "  -verbose: print additional details\n"
              + "  -segment X: only check the specified segments.  This can be specified multiple\n"
              + "              times, to check more than one segment, e.g. '-segment _2 -segment _a'.\n"
              + "              You can't use this with the -exorcise option\n"
              + "  -threadCount X: number of threads used to check index concurrently.\n"
              + "                  When not specified, this will default to the number of CPU cores.\n"
              + "                  When '-threadCount 1' is used, index checking will be performed sequentially.\n"
              + "  -dir-impl X: use a specific "
              + FSDirectory.class.getSimpleName()
              + " implementation. "
              + "If no package is specified the "
              + FSDirectory.class.getPackage().getName()
              + " package will be used.\n"
              + "CheckIndex only verifies file checksums as default.\n"
              + "Use -level with value of '2' or higher if you also want to check segment file contents.\n\n"
              + "**WARNING**: -exorcise *LOSES DATA*. This should only be used on an emergency basis as it will cause\n"
              + "documents (perhaps many) to be permanently removed from the index.  Always make\n"
              + "a backup copy of your index before running this!  Do not run this tool on an index\n"
              + "that is actively being written to.  You have been warned!\n"
              + "\n"
              + "Run without -exorcise, this tool will open the index, report version information\n"
              + "and report any exceptions it hits and what action it would take if -exorcise were\n"
              + "specified.  With -exorcise, this tool will remove any segments that have issues and\n"
              + "write a new segments_N file.  This means all documents contained in the affected\n"
              + "segments will be removed.\n"
              + "\n"
              + "This tool exits with exit code 1 if the index cannot be opened or has any\n"
              + "corruption, else 0.\n");
    }

    if (opts.onlySegments.isEmpty()) {
      opts.onlySegments = null;
    } else if (opts.doExorcise) {
      throw new IllegalArgumentException("ERROR: cannot specify both -exorcise and -segment");
    }

    return opts;
  }

  /**
   * Actually perform the index check
   *
   * @param opts The options to use for this check
   * @return 0 iff the index is clean, 1 otherwise
   */
  @SuppressForbidden(reason = "Thread sleep")
  public int doCheck(Options opts) throws IOException, InterruptedException {
    setLevel(opts.level);
    setInfoStream(opts.out, opts.verbose);
    // user provided thread count via command line argument, overriding the default with user
    // provided value
    if (opts.threadCount > 0) {
      setThreadCount(opts.threadCount);
    }

    Status result = checkIndex(opts.onlySegments);

    if (result.missingSegments) {
      return 1;
    }

    if (result.clean == false) {
      if (opts.doExorcise == false) {
        opts.out.println(
            "WARNING: would write new segments file, and "
                + result.totLoseDocCount
                + " documents would be lost, if -exorcise were specified\n");
      } else {
        opts.out.println("WARNING: " + result.totLoseDocCount + " documents will be lost\n");
        opts.out.println(
            "NOTE: will write new segments file in 5 seconds; this will remove "
                + result.totLoseDocCount
                + " docs from the index. YOU WILL LOSE DATA. THIS IS YOUR LAST CHANCE TO CTRL+C!");
        for (int s = 0; s < 5; s++) {
          Thread.sleep(1000);
          opts.out.println("  " + (5 - s) + "...");
        }
        opts.out.println("Writing...");
        exorciseIndex(result);
        opts.out.println("OK");
        opts.out.println(
            "Wrote new segments file \"" + result.newSegments.getSegmentsFileName() + "\"");
      }
    }
    opts.out.println();

    return result.clean ? 0 : 1;
  }

  private static Status.SoftDeletesStatus checkSoftDeletes(
      String softDeletesField,
      SegmentCommitInfo info,
      SegmentReader reader,
      PrintStream infoStream,
      boolean failFast)
      throws IOException {

    Status.SoftDeletesStatus status = new Status.SoftDeletesStatus();
    if (infoStream != null) infoStream.print("    test: check soft deletes.....");
    try {
      int softDeletes =
          PendingSoftDeletes.countSoftDeletes(
              FieldExistsQuery.getDocValuesDocIdSetIterator(softDeletesField, reader),
              reader.getLiveDocs());
      if (softDeletes != info.getSoftDelCount()) {
        throw new CheckIndexException(
            "actual soft deletes: " + softDeletes + " but expected: " + info.getSoftDelCount());
      }
    } catch (Exception e) {
      if (failFast) {
        throw IOUtils.rethrowAlways(e);
      }
      msg(infoStream, "ERROR [" + e.getMessage() + "]");
      status.error = e;
      if (infoStream != null) {
        e.printStackTrace(infoStream);
      }
    }

    return status;
  }

  private static double nsToSec(long ns) {
    return ns / (double) TimeUnit.SECONDS.toNanos(1);
  }

  /**
   * The marker RuntimeException used by CheckIndex APIs when index integrity failure is detected.
   */
  public static class CheckIndexException extends RuntimeException {

    /**
     * Constructs a new CheckIndexException with the error message
     *
     * @param message the detailed error message.
     */
    public CheckIndexException(String message) {
      super(message);
    }

    /**
     * Constructs a new CheckIndexException with the error message, and the root cause
     *
     * @param message the detailed error message.
     * @param cause the underlying cause.
     */
    public CheckIndexException(String message, Throwable cause) {
      super(message, cause);
    }
  }
}
