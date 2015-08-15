/**
 * Copyright 2013 Brigham Young University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.byu.nlp.classify;

import java.util.List;

import org.apache.commons.math3.random.RandomGenerator;

import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.stats.UniformDistribution;
import edu.byu.nlp.util.IntArrays;

public class UniformClassifier implements ProbabilisticClassifier {

    private final int numLabels;
    private final RandomGenerator rnd;
    
    public UniformClassifier(int numLabels, RandomGenerator rnd) {
        this.numLabels = numLabels;
        this.rnd = rnd;
    }

    /** {@inheritDoc} */
    @Override
    public int classify(SparseFeatureVector s) {
        return rnd.nextInt(numLabels);
    }

    /** {@inheritDoc} */
    @Override
    public edu.byu.nlp.stats.CategoricalDistribution given(SparseFeatureVector condition) {
        return new UniformDistribution(numLabels, rnd);
    }

    /** {@inheritDoc} */
    @Override
    public List<Integer> classifyNBest(int n, SparseFeatureVector s) {
      return IntArrays.asList(IntArrays.sequence(0, numLabels)).subList(0, n);
    }
    
}