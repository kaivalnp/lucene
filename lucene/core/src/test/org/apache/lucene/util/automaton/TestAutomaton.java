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
package org.apache.lucene.util.automaton;

import static org.apache.lucene.util.automaton.Operations.DEFAULT_DETERMINIZE_WORK_LIMIT;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.tests.util.automaton.AutomatonTestUtil;
import org.apache.lucene.tests.util.automaton.AutomatonTestUtil.RandomAcceptedStrings;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefBuilder;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.IntsRefBuilder;
import org.apache.lucene.util.UnicodeUtil;
import org.apache.lucene.util.fst.Util;

public class TestAutomaton extends LuceneTestCase {

  public void testBasic() throws Exception {
    Automaton a = new Automaton();
    int start = a.createState();
    int x = a.createState();
    int y = a.createState();
    int end = a.createState();
    a.setAccept(end, true);

    a.addTransition(start, x, 'a', 'a');
    a.addTransition(start, end, 'd', 'd');
    a.addTransition(x, y, 'b', 'b');
    a.addTransition(y, end, 'c', 'c');
    a.finishState();
  }

  public void testReduceBasic() throws Exception {
    Automaton a = new Automaton();
    int start = a.createState();
    int end = a.createState();
    a.setAccept(end, true);
    // Should collapse to a-b:
    a.addTransition(start, end, 'a', 'a');
    a.addTransition(start, end, 'b', 'b');
    a.addTransition(start, end, 'm', 'm');
    // Should collapse to x-y:
    a.addTransition(start, end, 'x', 'x');
    a.addTransition(start, end, 'y', 'y');

    a.finishState();
    assertEquals(3, a.getNumTransitions(start));
    Transition scratch = new Transition();
    a.initTransition(start, scratch);
    a.getNextTransition(scratch);
    assertEquals('a', scratch.min);
    assertEquals('b', scratch.max);
    a.getNextTransition(scratch);
    assertEquals('m', scratch.min);
    assertEquals('m', scratch.max);
    a.getNextTransition(scratch);
    assertEquals('x', scratch.min);
    assertEquals('y', scratch.max);
  }

  public void testSameLanguage() throws Exception {
    Automaton a1 = Automata.makeString("foobar");
    Automaton a2 =
        Operations.concatenate(List.of(Automata.makeString("foo"), Automata.makeString("bar")));
    assertTrue(AutomatonTestUtil.sameLanguage(a1, a2));
  }

  public void testCommonPrefixString() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeString("foobar"), Automata.makeAnyString()));
    AutomatonTestUtil.assertCleanDFA(a); // not minimal
    assertEquals("foobar", Operations.getCommonPrefix(a));
  }

  public void testCommonPrefixEmpty() throws Exception {
    assertEquals("", Operations.getCommonPrefix(Automata.makeEmpty()));
  }

  public void testCommonPrefixEmptyString() throws Exception {
    assertEquals("", Operations.getCommonPrefix(Automata.makeEmptyString()));
  }

  public void testCommonPrefixAny() throws Exception {
    assertEquals("", Operations.getCommonPrefix(Automata.makeAnyString()));
  }

  public void testCommonPrefixRange() throws Exception {
    assertEquals("", Operations.getCommonPrefix(Automata.makeCharRange('a', 'b')));
  }

  public void testAlternatives() throws Exception {
    Automaton a = Automata.makeChar('a');
    Automaton c = Automata.makeChar('c');
    assertEquals("", Operations.getCommonPrefix(Operations.union(List.of(a, c))));
  }

  public void testCommonPrefixLeadingWildcard() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeAnyChar(), Automata.makeString("boo")));
    AutomatonTestUtil.assertMinimalDFA(a);
    assertEquals("", Operations.getCommonPrefix(a));
  }

  public void testCommonPrefixTrailingWildcard() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeString("boo"), Automata.makeAnyChar()));
    AutomatonTestUtil.assertMinimalDFA(a);
    assertEquals("boo", Operations.getCommonPrefix(a));
  }

  public void testCommonPrefixLeadingKleenStar() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeAnyString(), Automata.makeString("boo")));
    AutomatonTestUtil.assertCleanNFA(a);
    assertEquals("", Operations.getCommonPrefix(a));
  }

  public void testCommonPrefixTrailingKleenStar() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeString("boo"), Automata.makeAnyString()));
    AutomatonTestUtil.assertCleanDFA(a); // not minimal
    assertEquals("boo", Operations.getCommonPrefix(a));
  }

  public void testCommonPrefixOptional() throws Exception {
    Automaton a = new Automaton();
    int init = a.createState();
    int fini = a.createState();
    a.setAccept(init, true);
    a.setAccept(fini, true);
    a.addTransition(init, fini, 'm');
    a.addTransition(fini, fini, 'm');
    a.finishState();
    assertEquals("", Operations.getCommonPrefix(a));
  }

  public void testCommonPrefixNFA() throws Exception {
    Automaton a = new Automaton();
    int init = a.createState();
    int medial = a.createState();
    int fini = a.createState();
    a.setAccept(fini, true);
    a.addTransition(init, medial, 'm');
    a.addTransition(init, fini, 'm');
    a.addTransition(medial, fini, 'o');
    a.finishState();
    assertEquals("m", Operations.getCommonPrefix(a));
  }

  public void testCommonPrefixNFAInfinite() throws Exception {
    Automaton a = new Automaton();
    int init = a.createState();
    int medial = a.createState();
    int fini = a.createState();
    a.setAccept(fini, true);
    a.addTransition(init, medial, 'm');
    a.addTransition(init, fini, 'm');
    a.addTransition(medial, fini, 'm');
    a.addTransition(fini, fini, 'm');
    a.finishState();
    assertEquals("m", Operations.getCommonPrefix(a));
  }

  public void testCommonPrefixUnicode() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeString("boo😂😂😂"), Automata.makeAnyChar()));
    assertEquals("boo😂😂😂", Operations.getCommonPrefix(a));
  }

  public void testConcatenate1() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeString("m"), Automata.makeAnyString()));
    AutomatonTestUtil.assertCleanDFA(a); // not minimal
    assertTrue(Operations.run(a, "m"));
    assertTrue(Operations.run(a, "me"));
    assertTrue(Operations.run(a, "me too"));
  }

  public void testConcatenate2() throws Exception {
    Automaton a =
        Operations.concatenate(
            Arrays.asList(
                Automata.makeString("m"),
                Automata.makeAnyString(),
                Automata.makeString("n"),
                Automata.makeAnyString()));
    AutomatonTestUtil.assertCleanNFA(a);
    a = Operations.determinize(a, DEFAULT_DETERMINIZE_WORK_LIMIT);
    assertTrue(Operations.run(a, "mn"));
    assertTrue(Operations.run(a, "mone"));
    assertFalse(Operations.run(a, "m"));
    assertFalse(AutomatonTestUtil.isFinite(a));
  }

  public void testUnion1() throws Exception {
    Automaton a =
        Operations.union(
            Arrays.asList(Automata.makeString("foobar"), Automata.makeString("barbaz")));
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(Operations.run(a, "foobar"));
    assertTrue(Operations.run(a, "barbaz"));

    assertMatches(a, "foobar", "barbaz");
  }

  public void testUnion2() throws Exception {
    Automaton a =
        Operations.union(
            Arrays.asList(
                Automata.makeString("foobar"),
                Automata.makeString(""),
                Automata.makeString("barbaz")));
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(Operations.run(a, "foobar"));
    assertTrue(Operations.run(a, "barbaz"));
    assertTrue(Operations.run(a, ""));

    assertMatches(a, "", "foobar", "barbaz");
  }

  public void testMinimizeSimple() throws Exception {
    Automaton a = Automata.makeString("foobar");
    Automaton aMin = MinimizationOperations.minimize(a, DEFAULT_DETERMINIZE_WORK_LIMIT);

    assertTrue(AutomatonTestUtil.sameLanguage(a, aMin));
  }

  public void testMinimize2() throws Exception {
    Automaton a =
        Operations.union(
            Arrays.asList(Automata.makeString("foobar"), Automata.makeString("boobar")));
    Automaton aMin = MinimizationOperations.minimize(a, DEFAULT_DETERMINIZE_WORK_LIMIT);
    assertTrue(
        AutomatonTestUtil.sameLanguage(
            Operations.determinize(Operations.removeDeadStates(a), DEFAULT_DETERMINIZE_WORK_LIMIT),
            aMin));
  }

  public void testReverse() throws Exception {
    Automaton a = Automata.makeString("foobar");
    Automaton ra = Operations.reverse(a);
    AutomatonTestUtil.assertMinimalDFA(a);
    Automaton a2 = Operations.reverse(ra);
    AutomatonTestUtil.assertMinimalDFA(a2);

    assertTrue(AutomatonTestUtil.sameLanguage(a, a2));
  }

  public void testOptional() throws Exception {
    Automaton a = Automata.makeString("foobar");
    Automaton a2 = Operations.optional(a);
    AutomatonTestUtil.assertMinimalDFA(a2);

    assertTrue(Operations.run(a, "foobar"));
    assertFalse(Operations.run(a, ""));
    assertTrue(Operations.run(a2, "foobar"));
    assertTrue(Operations.run(a2, ""));
  }

  public void testRepeatAny() throws Exception {
    Automaton a = Automata.makeString("zee");
    Automaton a2 = Operations.repeat(a);
    AutomatonTestUtil.assertMinimalDFA(a2);

    assertTrue(Operations.run(a2, ""));
    assertTrue(Operations.run(a2, "zee"));
    assertTrue(Operations.run(a2, "zeezee"));
    assertTrue(Operations.run(a2, "zeezeezee"));
  }

  public void testRepeatMin() throws Exception {
    Automaton a = Automata.makeString("zee");
    Automaton a2 = Operations.repeat(a, 2);
    AutomatonTestUtil.assertCleanDFA(a2); // not minimal

    assertFalse(Operations.run(a2, ""));
    assertFalse(Operations.run(a2, "zee"));
    assertTrue(Operations.run(a2, "zeezee"));
    assertTrue(Operations.run(a2, "zeezeezee"));
  }

  public void testRepeatMinMax1() throws Exception {
    Automaton a = Automata.makeString("zee");
    Automaton a2 = Operations.repeat(a, 0, 2);
    AutomatonTestUtil.assertMinimalDFA(a2);

    assertTrue(Operations.run(a2, ""));
    assertTrue(Operations.run(a2, "zee"));
    assertTrue(Operations.run(a2, "zeezee"));
    assertFalse(Operations.run(a2, "zeezeezee"));
  }

  public void testRepeatMinMax2() throws Exception {
    Automaton a = Automata.makeString("zee");
    Automaton a2 = Operations.repeat(a, 2, 4);
    AutomatonTestUtil.assertMinimalDFA(a2);

    assertFalse(Operations.run(a2, ""));
    assertFalse(Operations.run(a2, "zee"));
    assertTrue(Operations.run(a2, "zeezee"));
    assertTrue(Operations.run(a2, "zeezeezee"));
    assertTrue(Operations.run(a2, "zeezeezeezee"));
    assertFalse(Operations.run(a2, "zeezeezeezeezee"));
  }

  public void testComplement() throws Exception {
    Automaton a = Automata.makeString("zee");
    Automaton a2 = Operations.complement(a, DEFAULT_DETERMINIZE_WORK_LIMIT);
    AutomatonTestUtil.assertMinimalDFA(a2);

    assertTrue(Operations.run(a2, ""));
    assertFalse(Operations.run(a2, "zee"));
    assertTrue(Operations.run(a2, "zeezee"));
    assertTrue(Operations.run(a2, "zeezeezee"));
  }

  public void testInterval() throws Exception {
    Automaton a = Automata.makeDecimalInterval(17, 100, 3);
    AutomatonTestUtil.assertCleanDFA(a); // not minimal

    assertFalse(Operations.run(a, ""));
    assertTrue(Operations.run(a, "017"));
    assertTrue(Operations.run(a, "100"));
    assertTrue(Operations.run(a, "073"));
  }

  public void testCommonSuffix() throws Exception {
    Automaton a = new Automaton();
    int init = a.createState();
    int fini = a.createState();
    a.setAccept(init, true);
    a.setAccept(fini, true);
    a.addTransition(init, fini, 'm');
    a.addTransition(fini, fini, 'm');
    a.finishState();
    assertEquals(0, Operations.getCommonSuffixBytesRef(a).length);
  }

  public void testCommonSuffixEmpty() throws Exception {
    assertEquals(newBytesRef(), Operations.getCommonSuffixBytesRef(Automata.makeEmpty()));
  }

  public void testCommonSuffixEmptyString() throws Exception {
    assertEquals(newBytesRef(), Operations.getCommonSuffixBytesRef(Automata.makeEmptyString()));
  }

  public void testCommonSuffixTrailingWildcard() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeString("boo"), Automata.makeAnyChar()));
    AutomatonTestUtil.assertMinimalDFA(a);
    assertEquals(newBytesRef(), Operations.getCommonSuffixBytesRef(a));
  }

  public void testCommonSuffixLeadingKleenStar() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeAnyString(), Automata.makeString("boo")));
    AutomatonTestUtil.assertCleanNFA(a);
    assertEquals(newBytesRef("boo"), Operations.getCommonSuffixBytesRef(a));
  }

  public void testCommonSuffixTrailingKleenStar() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeString("boo"), Automata.makeAnyString()));
    AutomatonTestUtil.assertCleanDFA(a); // not minimal
    assertEquals(newBytesRef(), Operations.getCommonSuffixBytesRef(a));
  }

  public void testCommonSuffixUnicode() throws Exception {
    Automaton a =
        Operations.concatenate(List.of(Automata.makeAnyString(), Automata.makeString("boo😂😂😂")));
    AutomatonTestUtil.assertCleanNFA(a);
    Automaton binary = new UTF32ToUTF8().convert(a);
    assertEquals(newBytesRef("boo😂😂😂"), Operations.getCommonSuffixBytesRef(binary));
  }

  public void testReverseRandom1() throws Exception {
    int ITERS = atLeast(100);
    for (int i = 0; i < ITERS; i++) {
      Automaton a = AutomatonTestUtil.randomAutomaton(random());
      Automaton ra = Operations.reverse(a);
      Automaton rra = Operations.reverse(ra);
      assertTrue(
          AutomatonTestUtil.sameLanguage(
              Operations.determinize(Operations.removeDeadStates(a), Integer.MAX_VALUE),
              Operations.determinize(Operations.removeDeadStates(rra), Integer.MAX_VALUE)));
    }
  }

  public void testReverseRandom2() throws Exception {
    int ITERS = atLeast(100);
    for (int iter = 0; iter < ITERS; iter++) {
      // System.out.println("TEST: iter=" + iter);
      Automaton a = AutomatonTestUtil.randomAutomaton(random());
      if (random().nextBoolean()) {
        a = Operations.removeDeadStates(a);
      }
      Automaton ra = Operations.reverse(a);
      Automaton rda = Operations.determinize(ra, Integer.MAX_VALUE);

      if (Operations.isEmpty(a)) {
        assertTrue(Operations.isEmpty(rda));
        continue;
      }

      RandomAcceptedStrings ras = new RandomAcceptedStrings(a);

      for (int iter2 = 0; iter2 < 20; iter2++) {
        // Find string accepted by original automaton
        int[] s = ras.getRandomAcceptedString(random());

        // Reverse it
        for (int j = 0; j < s.length / 2; j++) {
          int x = s[j];
          s[j] = s[s.length - j - 1];
          s[s.length - j - 1] = x;
        }
        // System.out.println("TEST:   iter2=" + iter2 + " s=" + Arrays.toString(s));

        // Make sure reversed automaton accepts it
        assertTrue(Operations.run(rda, new IntsRef(s, 0, s.length)));
      }
    }
  }

  public void testAnyStringEmptyString() throws Exception {
    Automaton a = Automata.makeAnyString();
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(Operations.run(a, ""));
  }

  public void testBasicIsEmpty() throws Exception {
    Automaton a = new Automaton();
    a.createState();
    assertTrue(Operations.isEmpty(a));
  }

  public void testRemoveDeadTransitionsEmpty() throws Exception {
    Automaton a = Automata.makeEmpty();
    Automaton a2 = Operations.removeDeadStates(a);
    assertTrue(Operations.isEmpty(a2));
  }

  public void testInvalidAddTransition() throws Exception {
    Automaton a = new Automaton();
    int s1 = a.createState();
    int s2 = a.createState();
    a.addTransition(s1, s2, 'a');
    a.addTransition(s2, s2, 'a');
    expectThrows(
        IllegalStateException.class,
        () -> {
          a.addTransition(s1, s2, 'b');
        });
  }

  public void testBuilderRandom() throws Exception {
    int ITERS = atLeast(100);
    for (int iter = 0; iter < ITERS; iter++) {
      Automaton a = AutomatonTestUtil.randomAutomaton(random());

      // Just get all transitions, shuffle, and build a new automaton with the same transitions:
      List<Transition> allTrans = new ArrayList<>();
      int numStates = a.getNumStates();
      for (int s = 0; s < numStates; s++) {
        int count = a.getNumTransitions(s);
        for (int i = 0; i < count; i++) {
          Transition t = new Transition();
          a.getTransition(s, i, t);
          allTrans.add(t);
        }
      }

      Automaton.Builder builder = new Automaton.Builder();
      for (int i = 0; i < numStates; i++) {
        int s = builder.createState();
        builder.setAccept(s, a.isAccept(s));
      }

      Collections.shuffle(allTrans, random());
      for (Transition t : allTrans) {
        builder.addTransition(t.source, t.dest, t.min, t.max);
      }

      assertTrue(
          AutomatonTestUtil.sameLanguage(
              Operations.determinize(Operations.removeDeadStates(a), Integer.MAX_VALUE),
              Operations.determinize(
                  Operations.removeDeadStates(builder.finish()), Integer.MAX_VALUE)));
    }
  }

  public void testIsTotal() throws Exception {
    assertFalse(Operations.isTotal(new Automaton()));
    Automaton a = new Automaton();
    int init = a.createState();
    int fini = a.createState();
    a.setAccept(fini, true);
    a.addTransition(init, fini, Character.MIN_CODE_POINT, Character.MAX_CODE_POINT);
    a.finishState();
    assertFalse(Operations.isTotal(a));
    a.addTransition(fini, fini, Character.MIN_CODE_POINT, Character.MAX_CODE_POINT);
    a.finishState();
    assertFalse(Operations.isTotal(a));
    a.setAccept(init, true);
    assertTrue(
        Operations.isTotal(MinimizationOperations.minimize(a, DEFAULT_DETERMINIZE_WORK_LIMIT)));
  }

  public void testMinimizeEmpty() throws Exception {
    Automaton a = new Automaton();
    int init = a.createState();
    int fini = a.createState();
    a.addTransition(init, fini, 'a');
    a.finishState();
    a = MinimizationOperations.minimize(a, DEFAULT_DETERMINIZE_WORK_LIMIT);
    assertEquals(0, a.getNumStates());
  }

  public void testMinus() throws Exception {
    Automaton a1 = Automata.makeString("foobar");
    Automaton a2 = Automata.makeString("boobar");
    Automaton a3 = Automata.makeString("beebar");
    Automaton a = Operations.union(Arrays.asList(a1, a2, a3));
    AutomatonTestUtil.assertCleanNFA(a); // this is why we have makeStringUnion()
    assertMatches(a, "foobar", "beebar", "boobar");

    Automaton a4 = Operations.minus(a, a2, DEFAULT_DETERMINIZE_WORK_LIMIT);
    AutomatonTestUtil.assertCleanDFA(a4);

    assertTrue(Operations.run(a4, "foobar"));
    assertFalse(Operations.run(a4, "boobar"));
    assertTrue(Operations.run(a4, "beebar"));
    assertMatches(a4, "foobar", "beebar");

    a4 = Operations.minus(a4, a1, DEFAULT_DETERMINIZE_WORK_LIMIT);
    AutomatonTestUtil.assertCleanDFA(a4);

    assertFalse(Operations.run(a4, "foobar"));
    assertFalse(Operations.run(a4, "boobar"));
    assertTrue(Operations.run(a4, "beebar"));
    assertMatches(a4, "beebar");

    a4 = Operations.minus(a4, a3, DEFAULT_DETERMINIZE_WORK_LIMIT);
    AutomatonTestUtil.assertCleanDFA(a4);

    assertFalse(Operations.run(a4, "foobar"));
    assertFalse(Operations.run(a4, "boobar"));
    assertFalse(Operations.run(a4, "beebar"));
    assertMatches(a4);
  }

  public void testOneInterval() throws Exception {
    Automaton a = Automata.makeDecimalInterval(999, 1032, 0);
    AutomatonTestUtil.assertCleanNFA(a);
    a = Operations.determinize(a, DEFAULT_DETERMINIZE_WORK_LIMIT);
    assertTrue(Operations.run(a, "0999"));
    assertTrue(Operations.run(a, "00999"));
    assertTrue(Operations.run(a, "000999"));
  }

  public void testAnotherInterval() throws Exception {
    Automaton a = Automata.makeDecimalInterval(1, 2, 0);
    AutomatonTestUtil.assertCleanDFA(a);
    assertTrue(Operations.run(a, "01"));
  }

  public void testIntervalRandom() throws Exception {
    int ITERS = atLeast(100);
    for (int iter = 0; iter < ITERS; iter++) {
      int min = TestUtil.nextInt(random(), 0, 100000);
      int max = TestUtil.nextInt(random(), min, min + 100000);
      int digits;
      if (random().nextBoolean()) {
        digits = 0;
      } else {
        String s = Integer.toString(max);
        digits = TestUtil.nextInt(random(), s.length(), 2 * s.length());
      }
      StringBuilder b = new StringBuilder();
      for (int i = 0; i < digits; i++) {
        b.append('0');
      }
      String prefix = b.toString();

      Automaton a =
          Operations.determinize(
              Automata.makeDecimalInterval(min, max, digits), DEFAULT_DETERMINIZE_WORK_LIMIT);
      if (random().nextBoolean()) {
        a = MinimizationOperations.minimize(a, DEFAULT_DETERMINIZE_WORK_LIMIT);
      }
      String mins = Integer.toString(min);
      String maxs = Integer.toString(max);
      if (digits > 0) {
        mins = prefix.substring(mins.length()) + mins;
        maxs = prefix.substring(maxs.length()) + maxs;
      }
      assertTrue(Operations.run(a, mins));
      assertTrue(Operations.run(a, maxs));

      for (int iter2 = 0; iter2 < 100; iter2++) {
        int x = random().nextInt(2 * max);
        boolean expected = x >= min && x <= max;
        String sx = Integer.toString(x);
        if (sx.length() < digits) {
          // Left-fill with 0s
          sx = b.substring(sx.length()) + sx;
        } else if (digits == 0) {
          // Left-fill with random number of 0s:
          int numZeros = random().nextInt(10);
          StringBuilder sb = new StringBuilder();
          for (int i = 0; i < numZeros; i++) {
            sb.append('0');
          }
          sb.append(sx);
          sx = sb.toString();
        }
        assertEquals(expected, Operations.run(a, sx));
      }
    }
  }

  private void assertMatches(Automaton a, String... strings) {
    Set<IntsRef> expected = new HashSet<>();
    for (String s : strings) {
      IntsRefBuilder ints = new IntsRefBuilder();
      expected.add(Util.toUTF32(s, ints));
    }

    assertEquals(expected, TestOperations.getFiniteStrings(a));
  }

  public void testConcatenatePreservesDet() throws Exception {
    Automaton a1 = Automata.makeString("foobar");
    AutomatonTestUtil.assertMinimalDFA(a1);
    Automaton a2 = Automata.makeString("baz");
    AutomatonTestUtil.assertMinimalDFA(a2);
    Automaton a3 = Operations.concatenate(Arrays.asList(a1, a2));
    AutomatonTestUtil.assertMinimalDFA(a3);
  }

  public void testRemoveDeadStates() throws Exception {
    Automaton a = new Automaton();
    int s1 = a.createState();
    a.createState(); // create dead state
    a.setAccept(s1, true);
    a.finishState();
    assertEquals(2, a.getNumStates());
    a = Operations.removeDeadStates(a);
    assertEquals(1, a.getNumStates());
  }

  public void testRemoveDeadStatesEmpty1() throws Exception {
    Automaton a = new Automaton();
    a.finishState();
    assertTrue(Operations.isEmpty(a));
    assertTrue(Operations.isEmpty(Operations.removeDeadStates(a)));
  }

  public void testRemoveDeadStatesEmpty2() throws Exception {
    Automaton a = new Automaton();
    a.finishState();
    assertTrue(Operations.isEmpty(a));
    assertTrue(Operations.isEmpty(Operations.removeDeadStates(a)));
  }

  public void testRemoveDeadStatesEmpty3() throws Exception {
    Automaton a = new Automaton();
    int init = a.createState();
    int fini = a.createState();
    a.addTransition(init, fini, 'a');
    Automaton a2 = Operations.removeDeadStates(a);
    assertEquals(0, a2.getNumStates());
  }

  public void testConcatEmpty() throws Exception {
    // If you concat empty automaton to anything the result should still be empty:
    Automaton a = Operations.concatenate(List.of(Automata.makeEmpty(), Automata.makeString("foo")));
    AutomatonTestUtil.assertMinimalDFA(a);
    assertEquals(new HashSet<>(), TestOperations.getFiniteStrings(a));

    a = Operations.concatenate(List.of(Automata.makeString("foo"), Automata.makeEmpty()));
    AutomatonTestUtil.assertMinimalDFA(a);
    assertEquals(new HashSet<>(), TestOperations.getFiniteStrings(a));
  }

  public void testSeemsNonEmptyButIsNot1() throws Exception {
    Automaton a = new Automaton();
    // Init state has a transition but doesn't lead to accept
    int init = a.createState();
    int s = a.createState();
    a.addTransition(init, s, 'a');
    a.finishState();
    assertTrue(Operations.isEmpty(a));
  }

  public void testSeemsNonEmptyButIsNot2() throws Exception {
    Automaton a = new Automaton();
    int init = a.createState();
    int s = a.createState();
    a.addTransition(init, s, 'a');
    // An orphan'd accept state
    s = a.createState();
    a.setAccept(s, true);
    a.finishState();
    assertTrue(Operations.isEmpty(a));
  }

  public void testSameLanguage1() throws Exception {
    Automaton a = Automata.makeEmptyString();
    Automaton a2 = Automata.makeEmptyString();
    int state = a2.createState();
    a2.addTransition(0, state, 'a');
    a2.finishState();
    assertTrue(AutomatonTestUtil.sameLanguage(a, Operations.removeDeadStates(a2)));
  }

  private Automaton randomNoOp(Automaton a) {
    switch (random().nextInt(7)) {
      case 0:
        if (VERBOSE) {
          System.out.println("  randomNoOp: determinize");
        }
        return Operations.determinize(a, Integer.MAX_VALUE);
      case 1:
        if (a.getNumStates() < 100) {
          if (VERBOSE) {
            System.out.println("  randomNoOp: minimize");
          }
          return MinimizationOperations.minimize(a, DEFAULT_DETERMINIZE_WORK_LIMIT);
        } else {
          if (VERBOSE) {
            System.out.println(
                "  randomNoOp: skip op=minimize: too many states (" + a.getNumStates() + ")");
          }
          return a;
        }
      case 2:
        if (VERBOSE) {
          System.out.println("  randomNoOp: removeDeadStates");
        }
        return Operations.removeDeadStates(a);
      case 3:
        if (VERBOSE) {
          System.out.println("  randomNoOp: reverse reverse");
        }
        a = Operations.reverse(a);
        a = randomNoOp(a);
        return Operations.reverse(a);
      case 4:
        if (VERBOSE) {
          System.out.println("  randomNoOp: concat empty string");
        }
        return Operations.concatenate(List.of(a, Automata.makeEmptyString()));
      case 5:
        if (VERBOSE) {
          System.out.println("  randomNoOp: union empty automaton");
        }
        return Operations.union(List.of(a, Automata.makeEmpty()));
      case 6:
        if (VERBOSE) {
          System.out.println("  randomNoOp: do nothing!");
        }
        return a;
    }
    assert false;
    return null;
  }

  private static boolean hasMassiveTerm(Collection<BytesRef> terms) {
    for (BytesRef term : terms) {
      if (term.length > Automata.MAX_STRING_UNION_TERM_LENGTH) {
        return true;
      }
    }
    return false;
  }

  private Automaton unionTerms(Collection<BytesRef> terms) {
    Automaton a;
    if (random().nextBoolean() || hasMassiveTerm(terms)) {
      if (VERBOSE) {
        System.out.println("TEST: unionTerms: use union");
      }
      List<Automaton> as = new ArrayList<>();
      for (BytesRef term : terms) {
        as.add(Automata.makeString(term.utf8ToString()));
      }
      a = Operations.union(as);
    } else {
      if (VERBOSE) {
        System.out.println("TEST: unionTerms: use makeStringUnion");
      }
      List<BytesRef> termsList = new ArrayList<>(terms);
      Collections.sort(termsList);
      a = Automata.makeStringUnion(termsList);
    }

    return randomNoOp(a);
  }

  private String getRandomString() {
    // return TestUtil.randomSimpleString(random());
    return TestUtil.randomRealisticUnicodeString(random());
  }

  public void testRandomFinite() throws Exception {

    int numTerms = atLeast(10);
    int iters = atLeast(100);

    if (VERBOSE) {
      System.out.println("TEST: numTerms=" + numTerms + " iters=" + iters);
    }

    Set<BytesRef> terms = new HashSet<>();
    while (terms.size() < numTerms) {
      terms.add(newBytesRef(getRandomString()));
    }

    Automaton a = unionTerms(terms);
    assertSame(terms, a);

    for (int iter = 0; iter < iters; iter++) {
      if (VERBOSE) {
        System.out.println(
            "TEST: iter="
                + iter
                + " numTerms="
                + terms.size()
                + " a.numStates="
                + a.getNumStates());
        /*
        System.out.println("  terms:");
        for(BytesRef term : terms) {
          System.out.println("    " + term);
        }
        */
      }
      switch (random().nextInt(15)) {
        case 0:
          // concatenate prefix
          {
            if (VERBOSE) {
              System.out.println("  op=concat prefix");
            }
            Set<BytesRef> newTerms = new HashSet<>();
            BytesRef prefix = newBytesRef(getRandomString());
            BytesRefBuilder newTerm = new BytesRefBuilder();
            for (BytesRef term : terms) {
              newTerm.copyBytes(prefix);
              newTerm.append(term);
              newTerms.add(newTerm.toBytesRef());
            }
            terms = newTerms;
            boolean wasDeterministic = a.isDeterministic();
            a = Operations.concatenate(List.of(Automata.makeString(prefix.utf8ToString()), a));
            if (wasDeterministic) {
              assertEquals(wasDeterministic, a.isDeterministic());
            }
          }
          break;

        case 1:
          // concatenate suffix
          {
            BytesRef suffix = newBytesRef(getRandomString());
            if (VERBOSE) {
              System.out.println("  op=concat suffix " + suffix);
            }
            Set<BytesRef> newTerms = new HashSet<>();
            BytesRefBuilder newTerm = new BytesRefBuilder();
            for (BytesRef term : terms) {
              newTerm.copyBytes(term);
              newTerm.append(suffix);
              newTerms.add(newTerm.toBytesRef());
            }
            terms = newTerms;
            a = Operations.concatenate(List.of(a, Automata.makeString(suffix.utf8ToString())));
          }
          break;

        case 2:
          // determinize
          if (VERBOSE) {
            System.out.println("  op=determinize");
          }
          a = Operations.determinize(a, Integer.MAX_VALUE);
          assertTrue(a.isDeterministic());
          break;

        case 3:
          if (a.getNumStates() < 100) {
            if (VERBOSE) {
              System.out.println("  op=minimize");
            }
            // minimize
            a = MinimizationOperations.minimize(a, DEFAULT_DETERMINIZE_WORK_LIMIT);
          } else if (VERBOSE) {
            System.out.println("  skip op=minimize: too many states (" + a.getNumStates() + ")");
          }
          break;

        case 4:
          // union
          {
            if (VERBOSE) {
              System.out.println("  op=union");
            }
            Set<BytesRef> newTerms = new HashSet<>();
            int numNewTerms = random().nextInt(5);
            while (newTerms.size() < numNewTerms) {
              newTerms.add(newBytesRef(getRandomString()));
            }
            terms.addAll(newTerms);
            Automaton newA = unionTerms(newTerms);
            a = Operations.union(List.of(a, newA));
          }
          break;

        case 5:
          // optional
          {
            if (VERBOSE) {
              System.out.println("  op=optional");
            }
            // NOTE: This can add a dead state:
            a = Operations.optional(a);
            terms.add(newBytesRef());
          }
          break;

        case 6:
          // minus finite
          {
            if (VERBOSE) {
              System.out.println("  op=minus finite");
            }
            if (terms.size() > 0) {
              RandomAcceptedStrings rasl =
                  new RandomAcceptedStrings(Operations.removeDeadStates(a));
              Set<BytesRef> toRemove = new HashSet<>();
              int numToRemove = TestUtil.nextInt(random(), 1, (terms.size() + 1) / 2);
              while (toRemove.size() < numToRemove) {
                int[] ints = rasl.getRandomAcceptedString(random());
                BytesRef term = newBytesRef(UnicodeUtil.newString(ints, 0, ints.length));
                if (toRemove.contains(term) == false) {
                  toRemove.add(term);
                }
              }
              for (BytesRef term : toRemove) {
                boolean removed = terms.remove(term);
                assertTrue(removed);
              }
              Automaton a2 = unionTerms(toRemove);
              a = Operations.minus(a, a2, Integer.MAX_VALUE);
            }
          }
          break;

        case 7:
          {
            // minus infinite
            List<Automaton> as = new ArrayList<>();
            int count = TestUtil.nextInt(random(), 1, 5);
            Set<Integer> prefixes = new HashSet<>();
            while (prefixes.size() < count) {
              // prefix is a leading ascii byte; we remove <prefix>* from a
              int prefix = random().nextInt(128);
              prefixes.add(prefix);
            }

            if (VERBOSE) {
              System.out.println("  op=minus infinite prefixes=" + prefixes);
            }

            for (int prefix : prefixes) {
              // prefix is a leading ascii byte; we remove <prefix>* from a
              Automaton a2 = new Automaton();
              int init = a2.createState();
              int state = a2.createState();
              a2.addTransition(init, state, prefix);
              a2.setAccept(state, true);
              a2.addTransition(state, state, Character.MIN_CODE_POINT, Character.MAX_CODE_POINT);
              a2.finishState();
              as.add(a2);
              Iterator<BytesRef> it = terms.iterator();
              while (it.hasNext()) {
                BytesRef term = it.next();
                if (term.length > 0 && (term.bytes[term.offset] & 0xFF) == prefix) {
                  it.remove();
                }
              }
            }
            Automaton a2 = randomNoOp(Operations.union(as));
            a = Operations.minus(a, a2, DEFAULT_DETERMINIZE_WORK_LIMIT);
          }
          break;

        case 8:
          {
            int count = TestUtil.nextInt(random(), 10, 20);
            if (VERBOSE) {
              System.out.println("  op=intersect infinite count=" + count);
            }
            // intersect infinite
            List<Automaton> as = new ArrayList<>();

            Set<Integer> prefixes = new HashSet<>();
            while (prefixes.size() < count) {
              int prefix = random().nextInt(128);
              prefixes.add(prefix);
            }
            if (VERBOSE) {
              System.out.println("  prefixes=" + prefixes);
            }

            for (int prefix : prefixes.stream().mapToInt(Integer::intValue).toArray()) {
              // prefix is a leading ascii byte; we retain <prefix>* in a
              Automaton a2 = new Automaton();
              int init = a2.createState();
              int state = a2.createState();
              a2.addTransition(init, state, prefix);
              a2.setAccept(state, true);
              a2.addTransition(state, state, Character.MIN_CODE_POINT, Character.MAX_CODE_POINT);
              a2.finishState();
              as.add(a2);
              prefixes.add(prefix);
            }

            Automaton a2 = Operations.union(as);
            if (random().nextBoolean()) {
              a2 = Operations.determinize(a2, DEFAULT_DETERMINIZE_WORK_LIMIT);
            } else if (random().nextBoolean()) {
              a2 = MinimizationOperations.minimize(a2, DEFAULT_DETERMINIZE_WORK_LIMIT);
            }
            a = Operations.intersection(a, a2);

            Iterator<BytesRef> it = terms.iterator();
            while (it.hasNext()) {
              BytesRef term = it.next();
              if (term.length == 0 || prefixes.contains(term.bytes[term.offset] & 0xff) == false) {
                if (VERBOSE) {
                  System.out.println("  drop term=" + term);
                }
                it.remove();
              } else {
                if (VERBOSE) {
                  System.out.println("  keep term=" + term);
                }
              }
            }
          }
          break;

        case 9:
          // reverse
          {
            if (VERBOSE) {
              System.out.println("  op=reverse");
            }
            a = Operations.reverse(a);
            Set<BytesRef> newTerms = new HashSet<>();
            for (BytesRef term : terms) {
              newTerms.add(
                  newBytesRef(new StringBuilder(term.utf8ToString()).reverse().toString()));
            }
            terms = newTerms;
          }
          break;

        case 10:
          if (VERBOSE) {
            System.out.println("  op=randomNoOp");
          }
          a = randomNoOp(a);
          break;

        case 11:
          // interval
          {
            int min = random().nextInt(1000);
            int max = min + random().nextInt(50);
            // digits must be non-zero else we make cycle
            int digits = Integer.toString(max).length();
            if (VERBOSE) {
              System.out.println(
                  "  op=union interval min=" + min + " max=" + max + " digits=" + digits);
            }
            a = Operations.union(List.of(a, Automata.makeDecimalInterval(min, max, digits)));
            StringBuilder b = new StringBuilder();
            for (int i = 0; i < digits; i++) {
              b.append('0');
            }
            String prefix = b.toString();
            for (int i = min; i <= max; i++) {
              String s = Integer.toString(i);
              if (s.length() < digits) {
                // Left-fill with 0s
                s = prefix.substring(s.length()) + s;
              }
              terms.add(newBytesRef(s));
            }
          }
          break;

        case 12:
          if (VERBOSE) {
            System.out.println("  op=remove the empty string");
          }
          a = Operations.minus(a, Automata.makeEmptyString(), DEFAULT_DETERMINIZE_WORK_LIMIT);
          terms.remove(newBytesRef());
          break;

        case 13:
          if (VERBOSE) {
            System.out.println("  op=add the empty string");
          }
          a = Operations.union(List.of(a, Automata.makeEmptyString()));
          terms.add(newBytesRef());
          break;

        case 14:
          // Safety in case we are really unlucky w/ the dice:
          if (terms.size() <= numTerms * 3) {
            if (VERBOSE) {
              System.out.println("  op=concat finite automaton");
            }
            int count = random().nextBoolean() ? 2 : 3;
            Set<BytesRef> addTerms = new HashSet<>();
            while (addTerms.size() < count) {
              addTerms.add(newBytesRef(getRandomString()));
            }
            if (VERBOSE) {
              for (BytesRef term : addTerms) {
                System.out.println("    term=" + term);
              }
            }
            Automaton a2 = unionTerms(addTerms);
            Set<BytesRef> newTerms = new HashSet<>();
            if (random().nextBoolean()) {
              // suffix
              if (VERBOSE) {
                System.out.println("  do suffix");
              }
              a = Operations.concatenate(List.of(a, randomNoOp(a2)));
              BytesRefBuilder newTerm = new BytesRefBuilder();
              for (BytesRef term : terms) {
                for (BytesRef suffix : addTerms) {
                  newTerm.copyBytes(term);
                  newTerm.append(suffix);
                  newTerms.add(newTerm.toBytesRef());
                }
              }
            } else {
              // prefix
              if (VERBOSE) {
                System.out.println("  do prefix");
              }
              a = Operations.concatenate(List.of(randomNoOp(a2), a));
              BytesRefBuilder newTerm = new BytesRefBuilder();
              for (BytesRef term : terms) {
                for (BytesRef prefix : addTerms) {
                  newTerm.copyBytes(prefix);
                  newTerm.append(term);
                  newTerms.add(newTerm.toBytesRef());
                }
              }
            }

            terms = newTerms;
          }
          break;
        default:
          throw new AssertionError();
      }

      assertSame(terms, a);
      assertEquals(AutomatonTestUtil.isDeterministicSlow(a), a.isDeterministic());

      if (random().nextInt(10) == 7) {
        a = verifyTopoSort(a);
      }
    }

    assertSame(terms, a);
  }

  /**
   * Runs topo sort, verifies transitions then only "go forwards", and builds and returns new
   * automaton with those remapped toposorted states.
   */
  private Automaton verifyTopoSort(Automaton a) {
    int[] sorted = Operations.topoSortStates(a);
    // This can be < if we removed dead states:
    assertTrue(sorted.length <= a.getNumStates());
    Automaton a2 = new Automaton();
    int[] stateMap = new int[a.getNumStates()];
    Arrays.fill(stateMap, -1);
    Transition transition = new Transition();
    for (int state : sorted) {
      int newState = a2.createState();
      a2.setAccept(newState, a.isAccept(state));

      // Each state should only appear once in the sort:
      assertEquals(-1, stateMap[state]);
      stateMap[state] = newState;
    }

    // 2nd pass: add new transitions
    for (int state : sorted) {
      int count = a.initTransition(state, transition);
      for (int i = 0; i < count; i++) {
        a.getNextTransition(transition);
        assert stateMap[transition.dest] > stateMap[state];
        a2.addTransition(
            stateMap[state], stateMap[transition.dest], transition.min, transition.max);
      }
    }

    a2.finishState();
    return a2;
  }

  private void assertSame(Collection<BytesRef> terms, Automaton a) {

    try {
      assertTrue(AutomatonTestUtil.isFinite(a));
      assertFalse(Operations.isTotal(a));

      Automaton detA = Operations.determinize(a, DEFAULT_DETERMINIZE_WORK_LIMIT);

      // Make sure all terms are accepted:
      IntsRefBuilder scratch = new IntsRefBuilder();
      for (BytesRef term : terms) {
        Util.toIntsRef(term, scratch);
        assertTrue(
            "failed to accept term=" + term.utf8ToString(),
            Operations.run(detA, term.utf8ToString()));
      }

      // Use getFiniteStrings:
      Set<IntsRef> expected = new HashSet<>();
      for (BytesRef term : terms) {
        IntsRefBuilder intsRef = new IntsRefBuilder();
        Util.toUTF32(term.utf8ToString(), intsRef);
        expected.add(intsRef.toIntsRef());
      }
      Set<IntsRef> actual = TestOperations.getFiniteStrings(a);

      if (expected.equals(actual) == false) {
        System.out.println("FAILED:");
        for (IntsRef term : expected) {
          if (actual.contains(term) == false) {
            System.out.println("  term=" + term + " should be accepted but isn't");
          }
        }
        for (IntsRef term : actual) {
          if (expected.contains(term) == false) {
            System.out.println("  term=" + term + " is accepted but should not be");
          }
        }
        throw new AssertionError("mismatch");
      }

      // Use sameLanguage:
      Automaton a2 =
          Operations.removeDeadStates(Operations.determinize(unionTerms(terms), Integer.MAX_VALUE));
      assertTrue(
          AutomatonTestUtil.sameLanguage(
              a2, Operations.removeDeadStates(Operations.determinize(a, Integer.MAX_VALUE))));

      // Do same check, in UTF8 space
      Automaton utf8 = randomNoOp(new UTF32ToUTF8().convert(a));

      Set<IntsRef> expected2 = new HashSet<>();
      for (BytesRef term : terms) {
        IntsRefBuilder intsRef = new IntsRefBuilder();
        Util.toIntsRef(term, intsRef);
        expected2.add(intsRef.toIntsRef());
      }
      assertEquals(expected2, TestOperations.getFiniteStrings(utf8));
    } catch (AssertionError ae) {
      System.out.println("TEST: FAILED: not same");
      System.out.println("  terms (count=" + terms.size() + "):");
      for (BytesRef term : terms) {
        System.out.println("    " + term);
      }
      System.out.println("  automaton:");
      System.out.println(a.toDot());
      // a.writeDot("fail");
      throw ae;
    }
  }

  private boolean accepts(Automaton a, BytesRef b) {
    IntsRefBuilder intsBuilder = new IntsRefBuilder();
    Util.toIntsRef(b, intsBuilder);
    return Operations.run(a, intsBuilder.toIntsRef());
  }

  private Automaton makeBinaryInterval(
      BytesRef minTerm, boolean minInclusive, BytesRef maxTerm, boolean maxInclusive) {

    if (VERBOSE) {
      System.out.println(
          "TEST: minTerm="
              + minTerm
              + " minInclusive="
              + minInclusive
              + " maxTerm="
              + maxTerm
              + " maxInclusive="
              + maxInclusive);
    }

    Automaton a =
        Automata.makeBinaryInterval(
            minTerm, minInclusive,
            maxTerm, maxInclusive);

    Automaton minA = MinimizationOperations.minimize(a, Integer.MAX_VALUE);
    if (minA.getNumStates() != a.getNumStates()) {
      assertTrue(minA.getNumStates() < a.getNumStates());
      System.out.println("Original was not minimal:");
      System.out.println("Original:\n" + a.toDot());
      System.out.println("Minimized:\n" + minA.toDot());
      System.out.println("minTerm=" + minTerm + " minInclusive=" + minInclusive);
      System.out.println("maxTerm=" + maxTerm + " maxInclusive=" + maxInclusive);
      fail("automaton was not minimal");
    }

    if (VERBOSE) {
      System.out.println(a.toDot());
    }

    return a;
  }

  public void testMakeBinaryIntervalFiniteCasesBasic() throws Exception {
    // 0 (incl) - 00 (incl)
    byte[] zeros = new byte[3];
    Automaton a =
        makeBinaryInterval(newBytesRef(zeros, 0, 1), true, newBytesRef(zeros, 0, 2), true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(AutomatonTestUtil.isFinite(a));
    assertFalse(accepts(a, newBytesRef()));
    assertTrue(accepts(a, newBytesRef(zeros, 0, 1)));
    assertTrue(accepts(a, newBytesRef(zeros, 0, 2)));
    assertFalse(accepts(a, newBytesRef(zeros, 0, 3)));

    // '' (incl) - 00 (incl)
    a = makeBinaryInterval(newBytesRef(), true, newBytesRef(zeros, 0, 2), true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(AutomatonTestUtil.isFinite(a));
    assertTrue(accepts(a, newBytesRef()));
    assertTrue(accepts(a, newBytesRef(zeros, 0, 1)));
    assertTrue(accepts(a, newBytesRef(zeros, 0, 2)));
    assertFalse(accepts(a, newBytesRef(zeros, 0, 3)));

    // '' (excl) - 00 (incl)
    a = makeBinaryInterval(newBytesRef(), false, newBytesRef(zeros, 0, 2), true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(AutomatonTestUtil.isFinite(a));
    assertFalse(accepts(a, newBytesRef()));
    assertTrue(accepts(a, newBytesRef(zeros, 0, 1)));
    assertTrue(accepts(a, newBytesRef(zeros, 0, 2)));
    assertFalse(accepts(a, newBytesRef(zeros, 0, 3)));

    // 0 (excl) - 00 (incl)
    a = makeBinaryInterval(newBytesRef(zeros, 0, 1), false, newBytesRef(zeros, 0, 2), true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(AutomatonTestUtil.isFinite(a));
    assertFalse(accepts(a, newBytesRef()));
    assertFalse(accepts(a, newBytesRef(zeros, 0, 1)));
    assertTrue(accepts(a, newBytesRef(zeros, 0, 2)));
    assertFalse(accepts(a, newBytesRef(zeros, 0, 3)));

    // 0 (excl) - 00 (excl)
    a = makeBinaryInterval(newBytesRef(zeros, 0, 1), false, newBytesRef(zeros, 0, 2), false);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(AutomatonTestUtil.isFinite(a));
    assertFalse(accepts(a, newBytesRef()));
    assertFalse(accepts(a, newBytesRef(zeros, 0, 1)));
    assertFalse(accepts(a, newBytesRef(zeros, 0, 2)));
    assertFalse(accepts(a, newBytesRef(zeros, 0, 3)));
  }

  public void testMakeBinaryIntervalFiniteCasesRandom() throws Exception {
    int iters = atLeast(100);
    for (int iter = 0; iter < iters; iter++) {
      BytesRef prefix = newBytesRef(TestUtil.randomRealisticUnicodeString(random()));

      BytesRefBuilder b = new BytesRefBuilder();
      b.append(prefix);
      int numZeros = random().nextInt(10);
      for (int i = 0; i < numZeros; i++) {
        b.append((byte) 0);
      }
      BytesRef minTerm = b.get();

      b = new BytesRefBuilder();
      b.append(minTerm);
      numZeros = random().nextInt(10);
      for (int i = 0; i < numZeros; i++) {
        b.append((byte) 0);
      }
      BytesRef maxTerm = b.get();

      boolean minInclusive = random().nextBoolean();
      boolean maxInclusive = random().nextBoolean();
      Automaton a =
          makeBinaryInterval(
              minTerm, minInclusive,
              maxTerm, maxInclusive);
      assertTrue(AutomatonTestUtil.isFinite(a));
      int expectedCount = maxTerm.length - minTerm.length + 1;
      if (minInclusive == false) {
        expectedCount--;
      }
      if (maxInclusive == false) {
        expectedCount--;
      }

      if (expectedCount <= 0) {
        assertTrue(Operations.isEmpty(a));
        continue;
      } else {
        // Enumerate all finite strings and verify the count matches what we expect:
        assertEquals(expectedCount, TestOperations.getFiniteStrings(a, expectedCount).size());
      }

      b = new BytesRefBuilder();
      b.append(minTerm);
      if (minInclusive == false) {
        assertFalse(accepts(a, b.get()));
        b.append((byte) 0);
      }
      while (b.length() < maxTerm.length) {
        b.append((byte) 0);

        boolean expected;
        if (b.length() == maxTerm.length) {
          expected = maxInclusive;
        } else {
          expected = true;
        }
        assertEquals(expected, accepts(a, b.get()));
      }
    }
  }

  public void testMakeBinaryIntervalRandom() throws Exception {
    int iters = atLeast(100);
    for (int iter = 0; iter < iters; iter++) {
      BytesRef minTerm = TestUtil.randomBinaryTerm(random());
      boolean minInclusive = random().nextBoolean();
      BytesRef maxTerm = TestUtil.randomBinaryTerm(random());
      boolean maxInclusive = random().nextBoolean();

      Automaton a = makeBinaryInterval(minTerm, minInclusive, maxTerm, maxInclusive);

      for (int iter2 = 0; iter2 < 500; iter2++) {
        BytesRef term = TestUtil.randomBinaryTerm(random());
        int minCmp = minTerm.compareTo(term);
        int maxCmp = maxTerm.compareTo(term);

        boolean expected;
        if (minCmp > 0 || maxCmp < 0) {
          expected = false;
        } else if (minCmp == 0 && maxCmp == 0) {
          expected = minInclusive && maxInclusive;
        } else if (minCmp == 0) {
          expected = minInclusive;
        } else if (maxCmp == 0) {
          expected = maxInclusive;
        } else {
          expected = true;
        }

        if (VERBOSE) {
          System.out.println("  check term=" + term + " expected=" + expected);
        }
        IntsRefBuilder intsBuilder = new IntsRefBuilder();
        Util.toIntsRef(term, intsBuilder);
        assertEquals(expected, Operations.run(a, intsBuilder.toIntsRef()));
      }
    }
  }

  private static IntsRef intsRef(String s) {
    IntsRefBuilder intsBuilder = new IntsRefBuilder();
    Util.toIntsRef(newBytesRef(s), intsBuilder);
    return intsBuilder.toIntsRef();
  }

  public void testMakeBinaryIntervalBasic() throws Exception {
    Automaton a = Automata.makeBinaryInterval(newBytesRef("bar"), true, newBytesRef("foo"), true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(Operations.run(a, intsRef("bar")));
    assertTrue(Operations.run(a, intsRef("foo")));
    assertTrue(Operations.run(a, intsRef("beep")));
    assertFalse(Operations.run(a, intsRef("baq")));
    assertTrue(Operations.run(a, intsRef("bara")));
  }

  public void testMakeBinaryIntervalLowerBoundEmptyString() throws Exception {
    Automaton a = Automata.makeBinaryInterval(newBytesRef(""), true, newBytesRef("bar"), true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(Operations.run(a, intsRef("")));
    assertTrue(Operations.run(a, intsRef("a")));
    assertTrue(Operations.run(a, intsRef("bar")));
    assertFalse(Operations.run(a, intsRef("bara")));
    assertFalse(Operations.run(a, intsRef("baz")));

    a = Automata.makeBinaryInterval(newBytesRef(""), false, newBytesRef("bar"), true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertFalse(Operations.run(a, intsRef("")));
    assertTrue(Operations.run(a, intsRef("a")));
    assertTrue(Operations.run(a, intsRef("bar")));
    assertFalse(Operations.run(a, intsRef("bara")));
    assertFalse(Operations.run(a, intsRef("baz")));
  }

  public void testMakeBinaryIntervalEqual() throws Exception {
    Automaton a = Automata.makeBinaryInterval(newBytesRef("bar"), true, newBytesRef("bar"), true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(Operations.run(a, intsRef("bar")));
    assertTrue(AutomatonTestUtil.isFinite(a));
    assertEquals(1, TestOperations.getFiniteStrings(a).size());
  }

  public void testMakeBinaryIntervalCommonPrefix() throws Exception {
    Automaton a =
        Automata.makeBinaryInterval(newBytesRef("bar"), true, newBytesRef("barfoo"), true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertFalse(Operations.run(a, intsRef("bam")));
    assertTrue(Operations.run(a, intsRef("bar")));
    assertTrue(Operations.run(a, intsRef("bara")));
    assertTrue(Operations.run(a, intsRef("barf")));
    assertTrue(Operations.run(a, intsRef("barfo")));
    assertTrue(Operations.run(a, intsRef("barfoo")));
    assertTrue(Operations.run(a, intsRef("barfonz")));
    assertFalse(Operations.run(a, intsRef("barfop")));
    assertFalse(Operations.run(a, intsRef("barfoop")));
  }

  public void testMakeBinaryExceptEmpty() throws Exception {
    Automaton a = Automata.makeNonEmptyBinary();
    AutomatonTestUtil.assertMinimalDFA(a);
    assertFalse(Operations.run(a, intsRef("")));
    assertTrue(Operations.run(a, intsRef(TestUtil.randomRealisticUnicodeString(random(), 1, 10))));
  }

  public void testMakeBinaryIntervalOpenMax() throws Exception {
    Automaton a = Automata.makeBinaryInterval(newBytesRef("bar"), true, null, true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertFalse(Operations.run(a, intsRef("bam")));
    assertTrue(Operations.run(a, intsRef("bar")));
    assertTrue(Operations.run(a, intsRef("bara")));
    assertTrue(Operations.run(a, intsRef("barf")));
    assertTrue(Operations.run(a, intsRef("barfo")));
    assertTrue(Operations.run(a, intsRef("barfoo")));
    assertTrue(Operations.run(a, intsRef("barfonz")));
    assertTrue(Operations.run(a, intsRef("barfop")));
    assertTrue(Operations.run(a, intsRef("barfoop")));
    assertTrue(Operations.run(a, intsRef("zzz")));
  }

  public void testMakeBinaryIntervalOpenMaxZeroLengthMin() throws Exception {
    // when including min, automaton should accept "a"
    Automaton a = Automata.makeBinaryInterval(newBytesRef(""), true, null, true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(Operations.run(a, intsRef("")));
    assertTrue(Operations.run(a, intsRef("a")));
    assertTrue(Operations.run(a, intsRef("aaaaaa")));
    // excluding min should still accept "a"
    a = Automata.makeBinaryInterval(newBytesRef(""), false, null, true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertFalse(Operations.run(a, intsRef("")));
    assertTrue(Operations.run(a, intsRef("a")));
    assertTrue(Operations.run(a, intsRef("aaaaaa")));
  }

  public void testMakeBinaryIntervalOpenMin() throws Exception {
    Automaton a = Automata.makeBinaryInterval(null, true, newBytesRef("foo"), true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertFalse(Operations.run(a, intsRef("foz")));
    assertFalse(Operations.run(a, intsRef("zzz")));
    assertTrue(Operations.run(a, intsRef("foo")));
    assertTrue(Operations.run(a, intsRef("")));
    assertTrue(Operations.run(a, intsRef("a")));
    assertTrue(Operations.run(a, intsRef("aaa")));
    assertTrue(Operations.run(a, intsRef("bz")));
  }

  public void testMakeBinaryIntervalOpenBoth() throws Exception {
    Automaton a = Automata.makeBinaryInterval(null, true, null, true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(Operations.run(a, intsRef("foz")));
    assertTrue(Operations.run(a, intsRef("zzz")));
    assertTrue(Operations.run(a, intsRef("foo")));
    assertTrue(Operations.run(a, intsRef("")));
    assertTrue(Operations.run(a, intsRef("a")));
    assertTrue(Operations.run(a, intsRef("aaa")));
    assertTrue(Operations.run(a, intsRef("bz")));
  }

  public void testAcceptAllEmptyStringMin() throws Exception {
    Automaton a = Automata.makeBinaryInterval(newBytesRef(), true, null, true);
    AutomatonTestUtil.assertMinimalDFA(a);
    assertTrue(AutomatonTestUtil.sameLanguage(Automata.makeAnyBinary(), a));
  }

  private static IntsRef toIntsRef(String s) {
    IntsRefBuilder b = new IntsRefBuilder();
    for (int i = 0, cp = 0; i < s.length(); i += Character.charCount(cp)) {
      cp = s.codePointAt(i);
      b.append(cp);
    }

    return b.get();
  }

  public void testGetSingleton() {
    int iters = atLeast(10000);
    for (int iter = 0; iter < iters; iter++) {
      String s = TestUtil.randomRealisticUnicodeString(random());
      Automaton a = Automata.makeString(s);
      assertEquals(toIntsRef(s), Operations.getSingleton(a));
    }
  }

  public void testGetSingletonEmptyString() {
    Automaton a = new Automaton();
    int s = a.createState();
    a.setAccept(s, true);
    a.finishState();
    assertEquals(new IntsRef(), Operations.getSingleton(a));
  }

  public void testGetSingletonNothing() {
    Automaton a = new Automaton();
    a.createState();
    a.finishState();
    assertNull(Operations.getSingleton(a));
  }

  public void testGetSingletonTwo() {
    Automaton a = new Automaton();
    int s = a.createState();
    int x = a.createState();
    a.setAccept(x, true);
    a.addTransition(s, x, 55);
    int y = a.createState();
    a.setAccept(y, true);
    a.addTransition(s, y, 58);
    a.finishState();
    assertNull(Operations.getSingleton(a));
  }

  // LUCENE-9981
  public void testDeterminizeTooMuchEffort() {
    // make sure determinize properly aborts, relatively quickly, for this regexp:
    expectThrows(
        TooComplexToDeterminizeException.class,
        () -> {
          Automaton a = new RegExp("(.*a){2000}").toAutomaton();
          Operations.determinize(a, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT);
        });
    // ... and for its reversed form too:
    expectThrows(
        TooComplexToDeterminizeException.class,
        () -> {
          Automaton a = new RegExp("(.*a){2000}").toAutomaton();
          a = Operations.reverse(a);
          Operations.determinize(a, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT);
        });
  }

  public void testMakeCharSetEmpty() {
    Automaton expected = Automata.makeEmpty();
    Automaton actual = Automata.makeCharSet(new int[] {});
    AutomatonTestUtil.assertMinimalDFA(actual);
    assertTrue(AutomatonTestUtil.sameLanguage(expected, actual));
  }

  public void testMakeCharSetOne() {
    Automaton expected = Automata.makeChar('a');
    Automaton actual = Automata.makeCharSet(new int[] {'a'});
    AutomatonTestUtil.assertMinimalDFA(actual);
    assertTrue(AutomatonTestUtil.sameLanguage(expected, actual));
  }

  public void testMakeCharSetTwo() {
    Automaton expected = Operations.union(List.of(Automata.makeChar('a'), Automata.makeChar('A')));
    Automaton actual = Automata.makeCharSet(new int[] {'a', 'A'});
    AutomatonTestUtil.assertMinimalDFA(actual);
    assertTrue(AutomatonTestUtil.sameLanguage(expected, actual));
  }

  public void testMakeCharSetDups() {
    Automaton expected = Automata.makeChar('a');
    Automaton actual = Automata.makeCharSet(new int[] {'a', 'a', 'a'});
    AutomatonTestUtil.assertMinimalDFA(actual);
    assertTrue(AutomatonTestUtil.sameLanguage(expected, actual));
  }
}
