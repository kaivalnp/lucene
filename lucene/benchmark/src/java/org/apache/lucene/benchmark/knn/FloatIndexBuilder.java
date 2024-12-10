package org.apache.lucene.benchmark.knn;

import static org.apache.lucene.benchmark.knn.RecallTool.readFromFile;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.VectorSimilarityFunction;
import picocli.CommandLine;

@CommandLine.Command(name = "build-float-index")
public class FloatIndexBuilder extends RecallTool.IndexBuilder {
  @CommandLine.Option(
      names = {"-ip", "--inputPath"},
      description = "input path of vectors",
      required = true)
  Path inputPath;

  @CommandLine.Option(
      names = {"-kf", "--knnField"},
      description = "knn field indexed as float vectors",
      required = true)
  String knnField;

  @CommandLine.Option(
      names = {"-f", "--function"},
      description = "vector similarity function for KNN field")
  VectorSimilarityFunction function = VectorSimilarityFunction.DOT_PRODUCT;

  @CommandLine.Option(
      names = {"-m", "--maxConn"},
      description = "maxConn of HNSW graph",
      required = true)
  int maxConn;

  @CommandLine.Option(
      names = {"-b", "--beamWidth"},
      description = "beamWidth of HNSW graph",
      required = true)
  int beamWidth;

  @CommandLine.Option(
      names = {"-n", "--numMergeWorkers"},
      description = "number of merge workers of HNSW graph",
      required = true)
  int numMergeWorkers;

  @Override
  Codec getCodec() {
    return new Lucene101Codec() {
      @Override
      public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
        return new Lucene99HnswVectorsFormat(
            maxConn, beamWidth, numMergeWorkers, ForkJoinPool.commonPool());
      }
    };
  }

  @Override
  List<Document> getDocuments() {
    List<Document> documents = new ArrayList<>();
    List<float[]> vectors = readFromFile(inputPath);
    for (float[] vector : vectors) {
      Document document = new Document();
      document.add(new KnnFloatVectorField(knnField, vector, function));
      documents.add(document);
    }
    return documents;
  }
}
