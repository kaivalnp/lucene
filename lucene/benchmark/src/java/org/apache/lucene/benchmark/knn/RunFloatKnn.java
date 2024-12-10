package org.apache.lucene.benchmark.knn;

import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.SimpleCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;
import picocli.CommandLine;

public abstract class RunFloatKnn extends RecallTool.RunQueries<float[], Set<String>> {

  @CommandLine.Option(
      names = {"-kf", "--knnField"},
      description = "knn field indexed as float vectors",
      required = true)
  String knnField;

  @CommandLine.Option(
      names = {"-k", "--topK"},
      description = "topK for knn search",
      required = true)
  int topK;

  @CommandLine.Option(
      names = {"-id", "--idField"},
      description = "id field indexed as string")
  String idField = "docid";

  @Override
  CollectorManager<?, Set<String>> createManager() {
    return StringFieldCollector.createManager(idField);
  }

  @CommandLine.Command(name = "run-float-knn-exact")
  public static class Exact extends RunFloatKnn {
    @Override
    public Query getQuery(float[] target) {
      return new ExactKnnVectorQuery(knnField, target, topK);
    }

    public static class ExactKnnVectorQuery extends KnnFloatVectorQuery {
      public ExactKnnVectorQuery(String field, float[] target, int k) {
        super(field, target, k);
      }

      @Override
      protected TopDocs approximateSearch(
          LeafReaderContext context,
          Bits acceptDocs,
          int visitedLimit,
          KnnCollectorManager knnCollectorManager)
          throws IOException {
        @SuppressWarnings("resource")
        FloatVectorValues values = context.reader().getFloatVectorValues(field);
        DocIdSetIterator filtered =
            new FilteredDocIdSetIterator(values.iterator()) {
              @Override
              protected boolean match(int doc) {
                return acceptDocs == null || acceptDocs.get(doc);
              }
            };
        return exactSearch(context, filtered, null);
      }
    }
  }

  @CommandLine.Command(name = "run-float-knn-approx")
  public static class Approx extends RunFloatKnn {
    @Override
    public Query getQuery(float[] target) {
      return new KnnFloatVectorQuery(knnField, target, topK);
    }
  }

  public static class StringFieldCollector extends SimpleCollector {
    private final String field;
    private final Set<String> values;
    private int docBase;
    private SortedDocValues docValues;

    StringFieldCollector(String field) {
      this.field = field;
      this.values = new HashSet<>();
    }

    @Override
    @SuppressWarnings("resource")
    protected void doSetNextReader(LeafReaderContext context) throws IOException {
      docBase = context.docBase;
      docValues = context.reader().getSortedDocValues(field);
    }

    @Override
    public void collect(int doc) throws IOException {
      if (docValues != null && docValues.advanceExact(doc)) {
        values.add(docValues.lookupOrd(docValues.ordValue()).utf8ToString());
      } else {
        throw new RuntimeException(
            String.format(
                Locale.ROOT, "String field %s not found for doc %d", field, docBase + doc));
      }
    }

    @Override
    public ScoreMode scoreMode() {
      return ScoreMode.COMPLETE_NO_SCORES;
    }

    public static CollectorManager<StringFieldCollector, Set<String>> createManager(String field) {
      return new CollectorManager<>() {
        @Override
        public StringFieldCollector newCollector() {
          return new StringFieldCollector(field);
        }

        @Override
        public Set<String> reduce(Collection<StringFieldCollector> collectors) {
          Set<String> result = new HashSet<>();
          for (StringFieldCollector collector : collectors) {
            result.addAll(collector.values);
          }
          return result;
        }
      };
    }
  }
}
