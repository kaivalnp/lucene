package org.apache.lucene.search;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.VectorUtil;

public class ThresholdKnnFloatVectorQuery extends AbstractKnnVectorQuery {
  private final float[] target;
  private final float traversalThreshold, resultThreshold;
  private final int visitLimit;

  public ThresholdKnnFloatVectorQuery(
      String field,
      float[] target,
      float traversalThreshold,
      float resultThreshold,
      int visitLimit,
      Query filter) {
    super(field, Integer.MAX_VALUE, filter);
    this.target = VectorUtil.checkFinite(Objects.requireNonNull(target, "target"));
    this.traversalThreshold = traversalThreshold;
    this.resultThreshold = resultThreshold;
    this.visitLimit = visitLimit;
  }

  @Override
  @SuppressWarnings("resource")
  protected TopDocs approximateSearch(LeafReaderContext context, Bits acceptDocs, int visitedLimit)
      throws IOException {
    ThresholdKnnCollector thresholdKnnCollector =
        new ThresholdKnnCollector(traversalThreshold, resultThreshold, visitLimit);
    context.reader().searchNearestVectors(field, target, thresholdKnnCollector, acceptDocs);
    return thresholdKnnCollector.topDocs();
  }

  @Override
  VectorScorer createVectorScorer(LeafReaderContext context, FieldInfo fi) throws IOException {
    if (fi.getVectorEncoding() != VectorEncoding.FLOAT32) {
      return null;
    }
    return VectorScorer.create(context, fi, target);
  }

  @Override
  public String toString(String field) {
    return getClass().getSimpleName()
        + "[field="
        + this.field
        + " target=["
        + target[0]
        + ",...] traversalThreshold="
        + traversalThreshold
        + " resultThreshold="
        + resultThreshold
        + " visitLimit="
        + visitLimit
        + "]";
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    if (!super.equals(o)) return false;
    ThresholdKnnFloatVectorQuery that = (ThresholdKnnFloatVectorQuery) o;
    return Float.compare(that.traversalThreshold, traversalThreshold) == 0
        && Float.compare(that.resultThreshold, resultThreshold) == 0
        && Arrays.equals(target, that.target);
  }

  @Override
  public int hashCode() {
    int result = Objects.hash(super.hashCode(), traversalThreshold, resultThreshold);
    result = 31 * result + Arrays.hashCode(target);
    return result;
  }
}
