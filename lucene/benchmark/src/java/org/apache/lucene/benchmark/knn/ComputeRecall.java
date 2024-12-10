package org.apache.lucene.benchmark.knn;

import java.util.Set;
import picocli.CommandLine;

public class ComputeRecall {
  @CommandLine.Command(name = "compute-recall-from-strings")
  public static class FromSetOfStrings extends RecallTool.ConsumeResults<Set<String>> {
    @Override
    int common(Set<String> baseline, Set<String> candidate) {
      return (int) baseline.stream().filter(candidate::contains).count();
    }

    @Override
    int total(Set<String> baseline) {
      return baseline.size();
    }
  }

  @CommandLine.Command(name = "compute-recall-from-counts")
  public static class FromCounts extends RecallTool.ConsumeResults<Integer> {
    @Override
    int common(Integer baseline, Integer candidate) {
      return candidate;
    }

    @Override
    int total(Integer baseline) {
      return baseline;
    }
  }
}
