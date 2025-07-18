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
package org.apache.lucene.search;

import java.io.IOException;
import java.util.Objects;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.Terms;

/**
 * A {@link Query} that matches documents that contain either a {@link KnnFloatVectorField}, {@link
 * org.apache.lucene.document.KnnByteVectorField} or a field that indexes norms or doc values.
 */
public class FieldExistsQuery extends Query {
  private final String field;

  /** Create a query that will match that have a value for the given {@code field}. */
  public FieldExistsQuery(String field) {
    this.field = Objects.requireNonNull(field);
  }

  /**
   * Returns a {@link DocIdSetIterator} from the given field or null if the field doesn't exist in
   * the reader or if the reader has no doc values for the field.
   */
  public static DocIdSetIterator getDocValuesDocIdSetIterator(String field, LeafReader reader)
      throws IOException {
    FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(field);
    final DocIdSetIterator iterator;
    if (fieldInfo != null) {
      switch (fieldInfo.getDocValuesType()) {
        case NONE:
          iterator = null;
          break;
        case NUMERIC:
          iterator = reader.getNumericDocValues(field);
          break;
        case BINARY:
          iterator = reader.getBinaryDocValues(field);
          break;
        case SORTED:
          iterator = reader.getSortedDocValues(field);
          break;
        case SORTED_NUMERIC:
          iterator = reader.getSortedNumericDocValues(field);
          break;
        case SORTED_SET:
          iterator = reader.getSortedSetDocValues(field);
          break;
        default:
          throw new AssertionError();
      }
      return iterator;
    }
    return null;
  }

  public String getField() {
    return field;
  }

  @Override
  public String toString(String field) {
    return "FieldExistsQuery [field=" + this.field + "]";
  }

  @Override
  public void visit(QueryVisitor visitor) {
    if (visitor.acceptField(field)) {
      visitor.visitLeaf(this);
    }
  }

  @Override
  public boolean equals(Object other) {
    return sameClassAs(other) && field.equals(((FieldExistsQuery) other).field);
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int hash = classHash();
    hash = prime * hash + field.hashCode();
    return hash;
  }

  @Override
  public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    IndexReader reader = indexSearcher.getIndexReader();
    boolean allReadersRewritable = true;

    for (LeafReaderContext context : reader.leaves()) {
      LeafReader leaf = context.reader();
      FieldInfos fieldInfos = leaf.getFieldInfos();
      FieldInfo fieldInfo = fieldInfos.fieldInfo(field);

      if (fieldInfo == null) {
        allReadersRewritable = false;
        break;
      }

      if (fieldInfo.hasNorms()) { // the field indexes norms
        if (reader.getDocCount(field) != reader.maxDoc()) {
          allReadersRewritable = false;
          break;
        }
      } else if (fieldInfo.getVectorDimension() != 0) { // the field indexes vectors
        if (getVectorValuesSize(fieldInfo, leaf) != leaf.maxDoc()) {
          allReadersRewritable = false;
          break;
        }
      } else if (fieldInfo.getDocValuesType()
          != DocValuesType.NONE) { // the field indexes doc values or points

        // This optimization is possible due to LUCENE-9334 enforcing a field to always use the
        // same data structures (all or nothing).
        final Terms terms = leaf.terms(field);
        final PointValues pointValues = leaf.getPointValues(field);
        final DocValuesSkipper docValuesSkipper = leaf.getDocValuesSkipper(field);

        if ((terms == null || terms.getDocCount() != leaf.maxDoc())
            && (pointValues == null || pointValues.getDocCount() != leaf.maxDoc())
            && (docValuesSkipper == null || docValuesSkipper.docCount() != leaf.maxDoc())) {
          allReadersRewritable = false;
          break;
        }
      } else {
        throw new IllegalStateException(buildErrorMsg(fieldInfo));
      }
    }
    if (allReadersRewritable) {
      return new MatchAllDocsQuery();
    }
    return super.rewrite(indexSearcher);
  }

  @Override
  public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
    return new ConstantScoreWeight(this, boost) {

      @Override
      public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
        FieldInfos fieldInfos = context.reader().getFieldInfos();
        FieldInfo fieldInfo = fieldInfos.fieldInfo(field);
        DocIdSetIterator iterator = null;

        if (fieldInfo == null) {
          return null;
        }

        if (fieldInfo.hasNorms()) { // the field indexes norms
          iterator = context.reader().getNormValues(field);
        } else if (fieldInfo.getVectorDimension() != 0) { // the field indexes vectors
          iterator =
              switch (fieldInfo.getVectorEncoding()) {
                case FLOAT32 -> context.reader().getFloatVectorValues(field).iterator();
                case BYTE -> context.reader().getByteVectorValues(field).iterator();
              };
        } else if (fieldInfo.getDocValuesType()
            != DocValuesType.NONE) { // the field indexes doc values
          switch (fieldInfo.getDocValuesType()) {
            case NUMERIC:
              iterator = context.reader().getNumericDocValues(field);
              break;
            case BINARY:
              iterator = context.reader().getBinaryDocValues(field);
              break;
            case SORTED:
              iterator = context.reader().getSortedDocValues(field);
              break;
            case SORTED_NUMERIC:
              iterator = context.reader().getSortedNumericDocValues(field);
              break;
            case SORTED_SET:
              iterator = context.reader().getSortedSetDocValues(field);
              break;
            case NONE:
            default:
              throw new AssertionError();
          }
        } else {
          throw new IllegalStateException(buildErrorMsg(fieldInfo));
        }

        if (iterator == null) {
          return null;
        }
        final var scorer = new ConstantScoreScorer(score(), scoreMode, iterator);
        return new DefaultScorerSupplier(scorer);
      }

      @Override
      public int count(LeafReaderContext context) throws IOException {
        LeafReader reader = context.reader();
        FieldInfos fieldInfos = reader.getFieldInfos();
        FieldInfo fieldInfo = fieldInfos.fieldInfo(field);

        if (fieldInfo == null) {
          return 0;
        }

        if (fieldInfo.hasNorms()) { // the field indexes norms
          // If every field has a value then we can shortcut
          if (reader.getDocCount(field) == reader.maxDoc()) {
            return reader.numDocs();
          }

          return super.count(context);
        } else if (fieldInfo.hasVectorValues()) { // the field indexes vectors
          if (reader.hasDeletions() == false) {
            return getVectorValuesSize(fieldInfo, reader);
          }
          return super.count(context);
        } else if (fieldInfo.getDocValuesType()
            != DocValuesType.NONE) { // the field indexes doc values
          if (reader.hasDeletions() == false) {
            if (fieldInfo.getPointDimensionCount() > 0) {
              PointValues pointValues = reader.getPointValues(field);
              return pointValues == null ? 0 : pointValues.getDocCount();
            } else if (fieldInfo.getIndexOptions() != IndexOptions.NONE) {
              Terms terms = reader.terms(field);
              return terms == null ? 0 : terms.getDocCount();
            } else if (fieldInfo.docValuesSkipIndexType() != DocValuesSkipIndexType.NONE) {
              DocValuesSkipper docValuesSkipper = reader.getDocValuesSkipper(field);
              return docValuesSkipper == null ? 0 : docValuesSkipper.docCount();
            }
          }

          return super.count(context);
        } else {
          throw new IllegalStateException(buildErrorMsg(fieldInfo));
        }
      }

      @Override
      public boolean isCacheable(LeafReaderContext context) {
        FieldInfos fieldInfos = context.reader().getFieldInfos();
        FieldInfo fieldInfo = fieldInfos.fieldInfo(field);

        if (fieldInfo != null && fieldInfo.getDocValuesType() != DocValuesType.NONE) {
          return DocValues.isCacheable(context, field);
        }

        return true;
      }
    };
  }

  private String buildErrorMsg(FieldInfo fieldInfo) {
    return "FieldExistsQuery requires that the field indexes doc values, norms or vectors, but field '"
        + fieldInfo.name
        + "' exists and indexes neither of these data structures";
  }

  private int getVectorValuesSize(FieldInfo fi, LeafReader reader) throws IOException {
    assert fi.name.equals(field);
    return switch (fi.getVectorEncoding()) {
      case FLOAT32 -> {
        FloatVectorValues floatVectorValues = reader.getFloatVectorValues(field);
        assert floatVectorValues != null : "unexpected null float vector values";
        yield floatVectorValues.size();
      }
      case BYTE -> {
        ByteVectorValues byteVectorValues = reader.getByteVectorValues(field);
        assert byteVectorValues != null : "unexpected null byte vector values";
        yield byteVectorValues.size();
      }
    };
  }
}
