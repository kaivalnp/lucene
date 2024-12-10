package org.apache.lucene.benchmark.knn;

import static java.nio.file.StandardOpenOption.CREATE_NEW;
import static java.nio.file.StandardOpenOption.WRITE;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.store.FSDirectory;
import picocli.CommandLine;

@CommandLine.Command(
    mixinStandardHelpOptions = true,
    name = "recall-tool",
    scope = CommandLine.ScopeType.INHERIT,
    showAtFileInUsageHelp = true,
    showDefaultValues = true,
    subcommandsRepeatable = true,
    subcommands = {
      // Parse vectors
      ParseVectors.FromVec.class,

      // Build index
      FloatIndexBuilder.class,

      // Run queries
      RunFloatKnn.Exact.class,
      RunFloatKnn.Approx.class,
      RunFloatTnn.Exact.class,
      RunFloatTnn.Approx.class,

      // Consume results
      ComputeRecall.FromSetOfStrings.class,
      ComputeRecall.FromCounts.class
    })
public final class RecallTool {
  private RecallTool() {}

  public static void main(String[] args) {
    new CommandLine(RecallTool.class).execute(args);
  }

  public abstract static class ParseVectors<T> implements Runnable {
    @CommandLine.Option(
        names = {"-o", "--outputPath"},
        description = "output path of vectors",
        required = true)
    Path outputPath;

    public abstract List<T> getTargets();

    @Override
    public void run() {
      if (Files.isRegularFile(outputPath)) {
        List<T> queries = readFromFile(outputPath);
        System.out.printf(
            Locale.ROOT,
            "Vectors already exist at %s (count = %d), skipping\n",
            outputPath,
            queries.size());
      } else {
        List<T> queries = getTargets();
        writeToFile(outputPath, queries);
        System.out.printf(
            Locale.ROOT, "Vectors written to %s (count = %d)\n", outputPath, queries.size());
      }
    }
  }

  public abstract static class IndexBuilder implements Runnable {
    @CommandLine.Option(
        names = {"-i", "--indexPath"},
        description = "path of index",
        required = true)
    Path indexPath;

    @CommandLine.Option(
        names = {"-mns", "--maxNumSegments"},
        description = "max number of segments")
    int maxNumSegments = 1;

    abstract Codec getCodec();

    abstract List<Document> getDocuments();

    @Override
    public void run() {
      if (Files.isDirectory(indexPath)) {
        System.out.printf(Locale.ROOT, "Index already exists at %s, skipping", indexPath);
        return;
      }

      IndexWriterConfig config =
          new IndexWriterConfig()
              .setInfoStream(System.out)
              .setOpenMode(IndexWriterConfig.OpenMode.CREATE)
              .setCodec(getCodec())
              .setRAMBufferSizeMB(8192)
              .setUseCompoundFile(false);
      try (IndexWriter writer = new IndexWriter(FSDirectory.open(indexPath), config)) {
        for (Document document : getDocuments()) {
          writer.addDocument(document);
        }
        writer.forceMerge(maxNumSegments);
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
  }

  public record RunResult<R>(List<R> results, float avgLatency) implements Serializable {
    @Override
    public String toString() {
      return String.format(
          Locale.ROOT, "%d results, %.2f ms avg per query", results.size(), avgLatency);
    }
  }

  public abstract static class RunQueries<T, R> implements Runnable {
    @CommandLine.Option(
        names = {"-i", "--inputPath"},
        description = "input path of vectors",
        required = true)
    Path inputPath;

    @CommandLine.Option(
        names = {"-o", "--outputPath"},
        description = "output path of results",
        required = true)
    Path outputPath;

    @CommandLine.Option(
        names = {"-ip", "--indexPath"},
        description = "path of Lucene index",
        required = true)
    Path indexPath;

    abstract Query getQuery(T target);

    abstract CollectorManager<?, R> createManager();

    @Override
    public void run() {
      if (Files.isRegularFile(outputPath)) {
        RunResult<R> result = readFromFile(outputPath);
        System.out.printf(
            Locale.ROOT, "Results already exist at %s (%s), skipping\n", outputPath, result);
        return;
      }

      try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(indexPath))) {
        IndexSearcher searcher = new IndexSearcher(reader);
        CollectorManager<?, R> manager = createManager();

        LongAdder count = new LongAdder(), latency = new LongAdder();

        List<T> targets = readFromFile(inputPath);
        List<R> results =
            targets.parallelStream()
                .map(
                    target -> {
                      try {
                        count.increment();
                        System.out.printf(Locale.ROOT, "Running query %d\r", count.intValue());

                        long start = System.nanoTime();
                        R queryResults = searcher.search(getQuery(target), manager);
                        latency.add(System.nanoTime() - start);

                        return queryResults;
                      } catch (IOException e) {
                        throw new UncheckedIOException(e);
                      }
                    })
                .collect(Collectors.toList());
        float avgLatency = latency.floatValue() * 1e-6f / results.size();

        RunResult<R> result = new RunResult<>(results, avgLatency);
        writeToFile(outputPath, result);

        System.out.printf(Locale.ROOT, "%s written to %s\n", result, outputPath);
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
  }

  public abstract static class ConsumeResults<R> implements Runnable {
    @CommandLine.Option(
        names = {"-b", "--baseline"},
        description = "baseline results",
        required = true)
    Path baseline;

    @CommandLine.Option(
        names = {"-c", "--candidate"},
        description = "candidate results",
        required = true)
    Path candidate;

    abstract int common(R baseline, R candidate);

    abstract int total(R baseline);

    @Override
    public void run() {
      RunResult<R> baselineRunResult = readFromFile(baseline);
      RunResult<R> candidateRunResult = readFromFile(candidate);

      List<R> baselineResults = baselineRunResult.results;
      List<R> candidateResults = candidateRunResult.results;

      assert baselineResults.size() == candidateResults.size();

      long common = 0, total = 0;
      for (int i = 0; i < baselineResults.size(); i++) {
        common += common(baselineResults.get(i), candidateResults.get(i));
        total += total(baselineResults.get(i));
      }

      System.out.printf(
          Locale.ROOT,
          "Recall: %.3f\nBaseline Latency: %2.2f\nCandidate Latency: %2.2f",
          common / (float) total,
          baselineRunResult.avgLatency,
          candidateRunResult.avgLatency);
    }
  }

  @SuppressWarnings("unchecked")
  public static <T> T readFromFile(Path path) {
    try (ObjectInputStream ois =
        new ObjectInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
      return (T) ois.readObject();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  public static <T> void writeToFile(Path path, T object) {
    try (ObjectOutputStream oos =
        new ObjectOutputStream(
            new BufferedOutputStream(Files.newOutputStream(path, CREATE_NEW, WRITE)))) {
      oos.writeObject(object);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }
}
