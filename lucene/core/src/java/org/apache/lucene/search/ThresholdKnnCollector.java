package org.apache.lucene.search;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class ThresholdKnnCollector extends AbstractKnnCollector {
  private final float traversalThreshold, resultThreshold;
  private final List<ScoreDoc> scoreDocList;

  public ThresholdKnnCollector(float traversalThreshold, float resultThreshold, long visitLimit) {
    super(Integer.MAX_VALUE, visitLimit);
    this.traversalThreshold = traversalThreshold;
    this.resultThreshold = resultThreshold;
    this.scoreDocList = new ArrayList<>();
  }

  @Override
  public boolean collect(int docId, float similarity) {
    if (similarity >= resultThreshold) {
      return scoreDocList.add(new ScoreDoc(docId, similarity));
    }
    return false;
  }

  @Override
  public float minCompetitiveSimilarity() {
    return traversalThreshold;
  }

  @Override
  public TopDocs topDocs() {
    scoreDocList.sort(
        Comparator.<ScoreDoc, Float>comparing(scoreDoc -> -scoreDoc.score)
            .thenComparing(scoreDoc -> scoreDoc.doc));
    TotalHits.Relation relation =
        earlyTerminated()
            ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO
            : TotalHits.Relation.EQUAL_TO;
    return new TopDocs(
        new TotalHits(visitedCount(), relation), scoreDocList.toArray(ScoreDoc[]::new));
  }
}
