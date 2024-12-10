package org.apache.lucene.benchmark.knn;

import java.io.IOException;
import java.util.Collection;
import java.util.concurrent.atomic.LongAdder;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.SimpleCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;
import picocli.CommandLine;

public abstract class RunFloatTnn extends RecallTool.RunQueries<float[], Integer> {

  @CommandLine.Option(
      names = {"-kf", "--knnField"},
      description = "knn field indexed as float vectors",
      required = true)
  String knnField;

  @CommandLine.Option(
      names = {"-rs", "--resultSimilarity"},
      description = "result similarity for radius based search",
      required = true)
  float resultSimilarity;

  @Override
  CollectorManager<?, Integer> createManager() {
    return HitCountCollector.createManager();
  }

  @CommandLine.Command(name = "run-float-tnn-exact")
  public static class Exact extends RunFloatTnn {
    @Override
    public Query getQuery(float[] target) {
      return new ExactFloatVectorSimilarityQuery(knnField, target, resultSimilarity);
    }

    public static class ExactFloatVectorSimilarityQuery extends FloatVectorSimilarityQuery {
      private final float[] target;

      public ExactFloatVectorSimilarityQuery(String field, float[] target, float resultSimilarity) {
        super(field, target, resultSimilarity);
        this.target = target;
      }

      @Override
      protected TopDocs approximateSearch(
          LeafReaderContext context,
          Bits acceptDocs,
          int visitLimit,
          KnnCollectorManager knnCollectorManager)
          throws IOException {
        @SuppressWarnings("resource")
        VectorScorer scorer = context.reader().getFloatVectorValues(field).scorer(target);
        DocIdSetIterator iterator =
            new FilteredDocIdSetIterator(scorer.iterator()) {
              @Override
              protected boolean match(int doc) {
                return acceptDocs == null || acceptDocs.get(doc);
              }
            };

        int doc;
        KnnCollector collector = knnCollectorManager.newCollector(visitLimit, context);
        while ((doc = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
          collector.collect(doc, scorer.score());
        }

        return collector.topDocs();
      }
    }
  }

  @CommandLine.Command(name = "run-float-tnn-approx")
  public static class Approx extends RunFloatTnn {
    @CommandLine.Option(
        names = {"-ts", "--traversalSimilarity"},
        description = "traversal similarity for radius based search",
        required = true)
    float traversalSimilarity;

    @Override
    public Query getQuery(float[] target) {
      return new FloatVectorSimilarityQuery(
          knnField, target, traversalSimilarity, resultSimilarity);
    }
  }

  public static class HitCountCollector extends SimpleCollector {
    private final LongAdder adder;

    HitCountCollector() {
      this.adder = new LongAdder();
    }

    @Override
    public void collect(int doc) {
      adder.increment();
    }

    @Override
    public ScoreMode scoreMode() {
      return ScoreMode.COMPLETE_NO_SCORES;
    }

    public static CollectorManager<HitCountCollector, Integer> createManager() {
      return new CollectorManager<>() {
        @Override
        public HitCountCollector newCollector() {
          return new HitCountCollector();
        }

        @Override
        public Integer reduce(Collection<HitCountCollector> collectors) {
          LongAdder result = new LongAdder();
          for (HitCountCollector collector : collectors) {
            result.add(collector.adder.longValue());
          }
          return result.intValue();
        }
      };
    }
  }
}
