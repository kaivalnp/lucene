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
package org.apache.lucene.index;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.lucene.store.AlreadyClosedException;

/**
 * IndexReader is an abstract class, providing an interface for accessing a point-in-time view of an
 * index. Any changes made to the index via {@link IndexWriter} will not be visible until a new
 * {@code IndexReader} is opened. It's best to use {@link DirectoryReader#open(IndexWriter)} to
 * obtain an {@code IndexReader}, if your {@link IndexWriter} is in-process. When you need to
 * re-open to see changes to the index, it's best to use {@link
 * DirectoryReader#openIfChanged(DirectoryReader)} since the new reader will share resources with
 * the previous one when possible. Search of an index is done entirely through this abstract
 * interface, so that any subclass which implements it is searchable.
 *
 * <p>There are two different types of IndexReaders:
 *
 * <ul>
 *   <li>{@link LeafReader}: These indexes do not consist of several sub-readers, they are atomic.
 *       They support retrieval of stored fields, doc values, terms, and postings.
 *   <li>{@link CompositeReader}: Instances (like {@link DirectoryReader}) of this reader can only
 *       be used to get stored fields from the underlying LeafReaders, but it is not possible to
 *       directly retrieve postings. To do that, get the sub-readers via {@link
 *       CompositeReader#getSequentialSubReaders}.
 * </ul>
 *
 * <p>IndexReader instances for indexes on disk are usually constructed with a call to one of the
 * static <code>DirectoryReader.open()</code> methods, e.g. {@link
 * DirectoryReader#open(org.apache.lucene.store.Directory)}. {@link DirectoryReader} implements the
 * {@link CompositeReader} interface, it is not possible to directly get postings.
 *
 * <p>For efficiency, in this API documents are often referred to via <i>document numbers</i>,
 * non-negative integers which each name a unique document in the index. These document numbers are
 * ephemeral -- they may change as documents are added to and deleted from an index. Clients should
 * thus not rely on a given document having the same number between sessions.
 *
 * <p><a id="thread-safety"></a>
 *
 * <p><b>NOTE</b>: {@link IndexReader} instances are completely thread safe, meaning multiple
 * threads can call any of its methods, concurrently. If your application requires external
 * synchronization, you should <b>not</b> synchronize on the <code>IndexReader</code> instance; use
 * your own (non-Lucene) objects instead.
 */
public abstract sealed class IndexReader implements Closeable permits CompositeReader, LeafReader {

  private boolean closed = false;
  private boolean closedByChild = false;
  private final AtomicInteger refCount = new AtomicInteger(1);

  IndexReader() {}

  /**
   * A utility class that gives hooks in order to help build a cache based on the data that is
   * contained in this index.
   *
   * <p>Example: cache the number of documents that match a query per reader.
   *
   * <pre class="prettyprint">
   * public class QueryCountCache {
   *
   *   private final Query query;
   *   private final Map&lt;IndexReader.CacheKey, Integer&gt; counts = new ConcurrentHashMap&lt;&gt;();
   *
   *   // Create a cache of query counts for the given query
   *   public QueryCountCache(Query query) {
   *     this.query = query;
   *   }
   *
   *   // Count the number of matches of the query on the given IndexSearcher
   *   public int count(IndexSearcher searcher) throws IOException {
   *     IndexReader.CacheHelper cacheHelper = searcher.getIndexReader().getReaderCacheHelper();
   *     if (cacheHelper == null) {
   *       // reader doesn't support caching
   *       return searcher.count(query);
   *     } else {
   *       // make sure the cache entry is cleared when the reader is closed
   *       cacheHelper.addClosedListener(counts::remove);
   *       return counts.computeIfAbsent(cacheHelper.getKey(), cacheKey -&gt; {
   *         try {
   *           return searcher.count(query);
   *         } catch (IOException e) {
   *           throw new UncheckedIOException(e);
   *         }
   *       });
   *     }
   *   }
   *
   * }
   * </pre>
   *
   * @lucene.experimental
   */
  public interface CacheHelper {

    /**
     * Get a key that the resource can be cached on. The given entry can be compared using identity,
     * ie. {@link Object#equals} is implemented as {@code ==} and {@link Object#hashCode} is
     * implemented as {@link System#identityHashCode}.
     */
    CacheKey getKey();

    /**
     * Add a {@link ClosedListener} which will be called when the resource guarded by {@link
     * #getKey()} is closed.
     */
    void addClosedListener(ClosedListener listener);
  }

  /** A cache key identifying a resource that is being cached on. */
  public static final class CacheKey {
    CacheKey() {} // only instantiable by core impls
  }

  /**
   * A listener that is called when a resource gets closed.
   *
   * @lucene.experimental
   */
  @FunctionalInterface
  public interface ClosedListener {
    /**
     * Invoked when the resource (segment core, or index reader) that is being cached on is closed.
     */
    void onClose(CacheKey key) throws IOException;
  }

  /**
   * For test framework use only.
   *
   * @lucene.internal
   */
  protected void notifyReaderClosedListeners() throws IOException {
    // nothing to notify in the base impl
  }

  /** Expert: returns the current refCount for this reader */
  public final int getRefCount() {
    // NOTE: don't ensureOpen, so that callers can see
    // refCount is 0 (reader is closed)
    return refCount.get();
  }

  /**
   * Expert: increments the refCount of this IndexReader instance. RefCounts are used to determine
   * when a reader can be closed safely, i.e. as soon as there are no more references. Be sure to
   * always call a corresponding {@link #decRef}, in a finally clause; otherwise the reader may
   * never be closed. Note that {@link #close} simply calls decRef(), which means that the
   * IndexReader will not really be closed until {@link #decRef} has been called for all outstanding
   * references.
   *
   * @see #decRef
   * @see #tryIncRef
   */
  public final void incRef() {
    if (!tryIncRef()) {
      ensureOpen();
    }
  }

  /**
   * Expert: increments the refCount of this IndexReader instance only if the IndexReader has not
   * been closed yet and returns <code>true</code> iff the refCount was successfully incremented,
   * otherwise <code>false</code>. If this method returns <code>false</code> the reader is either
   * already closed or is currently being closed. Either way this reader instance shouldn't be used
   * by an application unless <code>true</code> is returned.
   *
   * <p>RefCounts are used to determine when a reader can be closed safely, i.e. as soon as there
   * are no more references. Be sure to always call a corresponding {@link #decRef}, in a finally
   * clause; otherwise the reader may never be closed. Note that {@link #close} simply calls
   * decRef(), which means that the IndexReader will not really be closed until {@link #decRef} has
   * been called for all outstanding references.
   *
   * @see #decRef
   * @see #incRef
   */
  public final boolean tryIncRef() {
    int count;
    while ((count = refCount.get()) > 0) {
      if (refCount.compareAndSet(count, count + 1)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Expert: decreases the refCount of this IndexReader instance. If the refCount drops to 0, then
   * this reader is closed. If an exception is hit, the refCount is unchanged.
   *
   * @throws IOException in case an IOException occurs in doClose()
   * @see #incRef
   */
  @SuppressWarnings("try")
  public final void decRef() throws IOException {
    // only check refcount here (don't call ensureOpen()), so we can
    // still close the reader if it was made invalid by a child:
    if (refCount.get() <= 0) {
      throw new AlreadyClosedException("this IndexReader is closed");
    }

    final int rc = refCount.decrementAndGet();
    if (rc == 0) {
      closed = true;
      try (Closeable _ = this::notifyReaderClosedListeners) {
        doClose();
      }
    } else if (rc < 0) {
      throw new IllegalStateException(
          "too many decRef calls: refCount is " + rc + " after decrement");
    }
  }

  /**
   * Throws AlreadyClosedException if this IndexReader or any of its child readers is closed,
   * otherwise returns.
   */
  protected final void ensureOpen() throws AlreadyClosedException {
    if (refCount.get() <= 0) {
      throw new AlreadyClosedException("this IndexReader is closed");
    }
    // the happens before rule on reading the refCount, which must be after the fake write,
    // ensures that we see the value:
    if (closedByChild) {
      throw new AlreadyClosedException(
          "this IndexReader cannot be used anymore as one of its child readers was closed");
    }
  }

  /**
   * {@inheritDoc}
   *
   * <p>{@code IndexReader} subclasses are not allowed to implement equals/hashCode, so methods are
   * declared final.
   */
  @Override
  public final boolean equals(Object obj) {
    return (this == obj);
  }

  /**
   * {@inheritDoc}
   *
   * <p>{@code IndexReader} subclasses are not allowed to implement equals/hashCode, so methods are
   * declared final.
   */
  @Override
  public final int hashCode() {
    return System.identityHashCode(this);
  }

  /**
   * Returns a {@link TermVectors} reader for the term vectors of this index.
   *
   * <p>This call never returns {@code null}, even if no term vectors were indexed. The returned
   * instance should only be used by a single thread.
   *
   * <p>Example:
   *
   * <pre class="prettyprint">
   * TopDocs hits = searcher.search(query, 10);
   * TermVectors termVectors = reader.termVectors();
   * for (ScoreDoc hit : hits.scoreDocs) {
   *   Fields vector = termVectors.get(hit.doc);
   * }
   * </pre>
   *
   * @throws IOException If there is a low-level IO error
   */
  public abstract TermVectors termVectors() throws IOException;

  /**
   * Returns the number of documents in this index.
   *
   * <p><b>NOTE</b>: This operation may run in O(maxDoc). Implementations that can't return this
   * number in constant-time should cache it.
   */
  public abstract int numDocs();

  /**
   * Returns one greater than the largest possible document number. This may be used to, e.g.,
   * determine how big to allocate an array which will have an element for every document number in
   * an index.
   */
  public abstract int maxDoc();

  /**
   * Returns the number of deleted documents.
   *
   * <p><b>NOTE</b>: This operation may run in O(maxDoc).
   */
  public final int numDeletedDocs() {
    return maxDoc() - numDocs();
  }

  /**
   * Returns a {@link StoredFields} reader for the stored fields of this index.
   *
   * <p>This call never returns {@code null}, even if no stored fields were indexed. The returned
   * instance should only be used by a single thread.
   *
   * <p>Example:
   *
   * <pre class="prettyprint">
   * TopDocs hits = searcher.search(query, 10);
   * StoredFields storedFields = reader.storedFields();
   * for (ScoreDoc hit : hits.scoreDocs) {
   *   Document doc = storedFields.document(hit.doc);
   * }
   * </pre>
   *
   * @throws IOException If there is a low-level IO error
   */
  public abstract StoredFields storedFields() throws IOException;

  /**
   * Returns true if any documents have been deleted. Implementers should consider overriding this
   * method if {@link #maxDoc()} or {@link #numDocs()} are not constant-time operations.
   */
  public boolean hasDeletions() {
    return numDeletedDocs() > 0;
  }

  /**
   * Closes files associated with this index. Also saves any new deletions to disk. No other methods
   * should be called after this has been called.
   *
   * @throws IOException if there is a low-level IO error
   */
  @Override
  public final synchronized void close() throws IOException {
    if (!closed) {
      decRef();
      closed = true;
    }
  }

  /** Implements close. */
  protected abstract void doClose() throws IOException;

  /**
   * Expert: Returns the root {@link IndexReaderContext} for this {@link IndexReader}'s sub-reader
   * tree.
   *
   * <p>Iff this reader is composed of sub readers, i.e. this reader being a composite reader, this
   * method returns a {@link CompositeReaderContext} holding the reader's direct children as well as
   * a view of the reader tree's atomic leaf contexts. All sub- {@link IndexReaderContext} instances
   * referenced from this readers top-level context are private to this reader and are not shared
   * with another context tree. For example, IndexSearcher uses this API to drive searching by one
   * atomic leaf reader at a time. If this reader is not composed of child readers, this method
   * returns an {@link LeafReaderContext}.
   *
   * <p>Note: Any of the sub-{@link CompositeReaderContext} instances referenced from this top-level
   * context do not support {@link CompositeReaderContext#leaves()}. Only the top-level context
   * maintains the convenience leaf-view for performance reasons.
   */
  public abstract IndexReaderContext getContext();

  /**
   * Returns the reader's leaves, or itself if this reader is atomic. This is a convenience method
   * calling {@code this.getContext().leaves()}.
   *
   * @see IndexReaderContext#leaves()
   */
  public final List<LeafReaderContext> leaves() {
    return getContext().leaves();
  }

  /**
   * Optional method: Return a {@link CacheHelper} that can be used to cache based on the content of
   * this reader. Two readers that have different data or different sets of deleted documents will
   * be considered different.
   *
   * <p>A return value of {@code null} indicates that this reader is not suited for caching, which
   * is typically the case for short-lived wrappers that alter the content of the wrapped reader.
   *
   * @lucene.experimental
   */
  public abstract CacheHelper getReaderCacheHelper();

  /**
   * Returns the number of documents containing the <code>term</code>. This method returns 0 if the
   * term or field does not exists. This method does not take into account deleted documents that
   * have not yet been merged away.
   *
   * @see TermsEnum#docFreq()
   */
  public abstract int docFreq(Term term) throws IOException;

  /**
   * Returns the total number of occurrences of {@code term} across all documents (the sum of the
   * freq() for each doc that has this term). Note that, like other term measures, this measure does
   * not take deleted documents into account.
   */
  public abstract long totalTermFreq(Term term) throws IOException;

  /**
   * Returns the sum of {@link TermsEnum#docFreq()} for all terms in this field. Note that, just
   * like other term measures, this measure does not take deleted documents into account.
   *
   * @see Terms#getSumDocFreq()
   */
  public abstract long getSumDocFreq(String field) throws IOException;

  /**
   * Returns the number of documents that have at least one term for this field. Note that, just
   * like other term measures, this measure does not take deleted documents into account.
   *
   * @see Terms#getDocCount()
   */
  public abstract int getDocCount(String field) throws IOException;

  /**
   * Returns the sum of {@link TermsEnum#totalTermFreq} for all terms in this field. Note that, just
   * like other term measures, this measure does not take deleted documents into account.
   *
   * @see Terms#getSumTotalTermFreq()
   */
  public abstract long getSumTotalTermFreq(String field) throws IOException;
}
