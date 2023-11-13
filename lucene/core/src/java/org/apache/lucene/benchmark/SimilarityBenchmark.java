package org.apache.lucene.benchmark;

import static java.nio.ByteOrder.LITTLE_ENDIAN;
import static java.nio.channels.FileChannel.MapMode.READ_ONLY;
import static java.nio.file.StandardOpenOption.READ;
import static org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;
import net.steppschuh.markdowngenerator.table.Table;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene95.Lucene95Codec;
import org.apache.lucene.codecs.lucene95.Lucene95HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import picocli.CommandLine;
import picocli.CommandLine.Option;

public class SimilarityBenchmark implements Runnable {
  @Option(
      names = {"--vecPath"},
      required = true)
  Path vecPath;

  @Option(names = {"--dim"})
  int dim = 100;

  @Option(names = {"--numDocs"})
  int numDocs = 1_000_000;

  @Option(names = {"--numQueries"})
  int numQueries = 10_000;

  @Option(
      names = {"--indexPath"},
      required = true)
  Path indexPath;

  @Option(names = {"--knnField"})
  String knnField = "knn";

  @Option(names = {"--function"})
  VectorSimilarityFunction function = DOT_PRODUCT;

  @Option(names = {"--maxNumSegments"})
  int maxNumSegments = 1;

  @Option(
      names = {"--maxConns"},
      split = ",")
  List<Integer> maxConns = List.of(16, 32, 64);

  @Option(
      names = {"--beamWidths"},
      split = ",")
  List<Integer> beamWidths = List.of(100, 200);

  @Option(
      names = {"--topKs"},
      split = ",")
  List<Integer> topKs = Collections.emptyList();

  @Option(
      names = {"--topK-thresholds"},
      split = ",")
  List<Float> topKThresholds = Collections.emptyList();

  @Option(
      names = {"--traversalSimilarities"},
      split = ",")
  List<Float> traversalSimilarities = Collections.emptyList();

  @Option(
      names = {"--resultSimilarities"},
      split = ",")
  List<Float> resultSimilarities = Collections.emptyList();

  public static void main(String... args) {
    new CommandLine(new SimilarityBenchmark()).execute(args);
  }

  @Override
  public void run() {
    assert topKs.size() == topKThresholds.size();
    assert traversalSimilarities.size() == resultSimilarities.size();

    float[][] docs = vectorsFromVecFile(numDocs, 0);
    createIndexes(docs);

    float[][] queries = vectorsFromVecFile(numQueries, numDocs);
    Map<Float, Result> cachedTrue = new HashMap<>();

    Table.Builder knnBuilder = new FloatFormattedBuilder();
    knnBuilder.addRow(
        "maxConn", "beamWidth", "topK", "threshold", "count", "numVisited", "latency", "recall");

    Table.Builder similarityBuilder = new FloatFormattedBuilder();
    similarityBuilder.addRow(
        "maxConn",
        "beamWidth",
        "traversalSimilarity",
        "resultSimilarity",
        "count",
        "numVisited",
        "latency",
        "recall");

    for (int maxConn : maxConns) {
      for (int beamWidth : beamWidths) {
        for (int index = 0; index < topKs.size(); index++) {
          Result result =
              knnResults(queries, maxConn, beamWidth, topKs.get(index), topKThresholds.get(index));
          Result baseline =
              cachedTrue.computeIfAbsent(
                  topKThresholds.get(index), t -> trueResults(queries, docs, t));
          knnBuilder.addRow(
              maxConn,
              beamWidth,
              topKs.get(index),
              topKThresholds.get(index),
              result.count,
              result.visited,
              result.time,
              result.count / baseline.count);
        }

        for (int index = 0; index < traversalSimilarities.size(); index++) {
          Result result =
              similarityResults(
                  queries,
                  maxConn,
                  beamWidth,
                  traversalSimilarities.get(index),
                  resultSimilarities.get(index));
          Result baseline =
              cachedTrue.computeIfAbsent(
                  resultSimilarities.get(index), t -> trueResults(queries, docs, t));
          similarityBuilder.addRow(
              maxConn,
              beamWidth,
              traversalSimilarities.get(index),
              resultSimilarities.get(index),
              result.count,
              result.visited,
              result.time,
              result.count / baseline.count);
        }
      }
    }

    System.out.printf(
        Locale.ROOT,
        "### KNN search\n\n%s\n\n### Similarity-based search\n\n%s\n",
        knnBuilder.build(),
        similarityBuilder.build());
  }

  private Path getIndexPath(int maxConn, int beamWidth) {
    return indexPath.resolve(
        String.format(
            Locale.ROOT,
            "%d-%d-%s-%s-%d-%d-%d",
            dim,
            numDocs,
            knnField,
            function,
            maxNumSegments,
            maxConn,
            beamWidth));
  }

  private Result trueResults(float[][] queries, float[][] docs, float threshold) {
    float similarity = (1 + threshold) / 2;

    LongAdder counts = new LongAdder();
    LongAdder numVisited = new LongAdder();
    LongAdder time = new LongAdder();

    Arrays.stream(queries)
        .parallel()
        .forEach(
            query -> {
              long startTime = System.currentTimeMillis();
              counts.add(
                  Arrays.stream(docs)
                      .parallel()
                      .filter(doc -> function.compare(query, doc) >= similarity)
                      .count());
              time.add(System.currentTimeMillis() - startTime);

              numVisited.add(docs.length);
            });

    return new Result(counts.longValue(), numVisited.longValue(), time.longValue(), numQueries);
  }

  private Result knnResults(
      float[][] queries, int maxConn, int beamWidth, int topK, float threshold) {
    Path resolvedIndexPath = getIndexPath(maxConn, beamWidth);
    try (Directory directory = FSDirectory.open(resolvedIndexPath);
        DirectoryReader reader = DirectoryReader.open(directory)) {

      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);

      float similarity = (1 + threshold) / 2;

      LongAdder counts = new LongAdder();
      LongAdder numVisited = new LongAdder();
      LongAdder time = new LongAdder();

      Arrays.stream(queries)
          .parallel()
          .forEach(
              query -> {
                try {
                  KnnFloatVectorQuery vectorQuery =
                      new KnnFloatVectorQuery(knnField, query, topK) {
                        @Override
                        protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
                          TopDocs merged = super.mergeLeafResults(perLeafResults);

                          int index =
                              -1
                                  - Arrays.binarySearch(
                                      merged.scoreDocs,
                                      new ScoreDoc(Integer.MAX_VALUE, similarity),
                                      Comparator.<ScoreDoc, Float>comparing(
                                              scoreDoc -> -scoreDoc.score)
                                          .thenComparing(scoreDoc -> scoreDoc.doc));

                          if (index < merged.scoreDocs.length) {
                            merged.scoreDocs = Arrays.copyOf(merged.scoreDocs, index);
                          }

                          numVisited.add(merged.totalHits.value);
                          return merged;
                        }
                      };

                  long startTime = System.currentTimeMillis();
                  counts.add(searcher.count(vectorQuery));
                  time.add(System.currentTimeMillis() - startTime);

                } catch (IOException e) {
                  throw new UncheckedIOException(e);
                }
              });

      return new Result(counts.longValue(), numVisited.longValue(), time.longValue(), numQueries);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private Result similarityResults(
      float[][] queries,
      int maxConn,
      int beamWidth,
      float traversalSimilarity,
      float resultSimilarity) {
    Path resolvedIndexPath = getIndexPath(maxConn, beamWidth);
    try (Directory directory = FSDirectory.open(resolvedIndexPath);
        DirectoryReader reader = DirectoryReader.open(directory)) {

      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);

      float traversalThreshold = (1 + traversalSimilarity) / 2;
      float resultThreshold = (1 + resultSimilarity) / 2;

      LongAdder counts = new LongAdder();
      LongAdder numVisited = new LongAdder();
      LongAdder time = new LongAdder();

      Arrays.stream(queries)
          .parallel()
          .forEach(
              query -> {
                try {
                  FloatVectorSimilarityQuery vectorQuery =
                      new FloatVectorSimilarityQuery(
                          knnField, query, traversalThreshold, resultThreshold, Long.MAX_VALUE) {
                        @Override
                        protected void approximateSearch(
                            LeafReaderContext context, KnnCollector collector) throws IOException {
                          super.approximateSearch(context, collector);
                          numVisited.add(collector.visitedCount());
                        }
                      };

                  long startTime = System.currentTimeMillis();
                  counts.add(searcher.count(vectorQuery));
                  time.add(System.currentTimeMillis() - startTime);

                } catch (IOException e) {
                  throw new UncheckedIOException(e);
                }
              });

      return new Result(counts.longValue(), numVisited.longValue(), time.longValue(), numQueries);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private float[][] vectorsFromVecFile(int numVectors, int skip) {
    try (FileChannel channel = FileChannel.open(vecPath, READ)) {
      float[][] results = new float[numVectors][dim];

      long position = (long) dim * skip * Float.BYTES;
      long size = (long) dim * numVectors * Float.BYTES;
      FloatBuffer floatBuffer =
          channel.map(READ_ONLY, position, size).order(LITTLE_ENDIAN).asFloatBuffer();

      for (float[] vector : results) {
        floatBuffer.get(vector);
      }

      return results;
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private void createIndexes(float[][] docs) {
    for (int maxConn : maxConns) {
      for (int beamWidth : beamWidths) {
        Path resolvedIndexPath = getIndexPath(maxConn, beamWidth);
        if (Files.isDirectory(resolvedIndexPath)) {
          continue;
        }

        IndexWriterConfig config =
            new IndexWriterConfig()
                .setCodec(
                    new Lucene95Codec() {
                      @Override
                      public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                        return new Lucene95HnswVectorsFormat(maxConn, beamWidth);
                      }
                    })
                .setRAMBufferSizeMB(4096)
                .setUseCompoundFile(false);

        try (Directory directory = FSDirectory.open(resolvedIndexPath);
            IndexWriter writer = new IndexWriter(directory, config)) {
          for (float[] vector : docs) {
            Document document = new Document();
            document.add(new KnnFloatVectorField(knnField, vector, function));
            writer.addDocument(document);
          }
          writer.forceMerge(maxNumSegments);
        } catch (IOException e) {
          throw new UncheckedIOException(e);
        }
      }
    }
  }

  private record Result(float count, float visited, float time) {
    public Result(long count, long visited, long time, float numQueries) {
      this(count / numQueries, visited / numQueries, time / numQueries);
    }
  }

  private static class FloatFormattedBuilder extends Table.Builder {
    @Override
    public Table.Builder addRow(Object... objects) {
      for (int index = 0; index < objects.length; index++) {
        if (objects[index] instanceof Float f) {
          objects[index] = String.format(Locale.ROOT, "%.2f", f);
        }
      }
      return super.addRow(objects);
    }
  }
}
