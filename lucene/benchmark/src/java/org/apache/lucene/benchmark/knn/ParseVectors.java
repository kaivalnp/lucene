package org.apache.lucene.benchmark.knn;

import static java.nio.ByteOrder.LITTLE_ENDIAN;
import static java.nio.channels.FileChannel.MapMode.READ_ONLY;
import static java.nio.file.StandardOpenOption.READ;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import picocli.CommandLine;

public abstract class ParseVectors extends RecallTool.ParseVectors<float[]> {
  @CommandLine.Option(
      names = {"-i", "--inputPath"},
      description = "path of file",
      required = true)
  Path inputPath;

  @CommandLine.Option(
      names = {"-s", "--skip"},
      description = "number of vectors to skip")
  int skip = 0;

  @CommandLine.Option(
      names = {"-n", "--num"},
      description = "maximum number of vectors")
  int num = Integer.MAX_VALUE;

  @CommandLine.Command(name = "generate-floats-from-vec")
  public static class FromVec extends ParseVectors {
    @CommandLine.Option(
        names = {"-d", "--dim"},
        description = "number of dimensions",
        required = true)
    int dim;

    @Override
    public List<float[]> getTargets() {
      try (FileChannel channel = FileChannel.open(inputPath, READ)) {
        long position = Math.min(channel.size(), (long) skip * dim * Float.BYTES);
        long size = Math.min(channel.size(), position + (long) num * dim * Float.BYTES) - position;
        FloatBuffer buffer =
            channel.map(READ_ONLY, position, size).order(LITTLE_ENDIAN).asFloatBuffer();

        List<float[]> results = new ArrayList<>();
        for (int i = 0; i < num && buffer.hasRemaining(); i++) {
          results.add(new float[dim]);
          buffer.get(results.getLast());
        }
        return results;
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
  }
}
