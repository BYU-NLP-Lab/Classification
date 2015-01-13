/**
 * Copyright 2015 Brigham Young University
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
package edu.byu.nlp.classify.eval;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.base.Preconditions;

import edu.byu.nlp.classify.eval.ProbabilisticLabelErrorFunction.CategoricalDistribution;
import edu.byu.nlp.stats.RandomGenerators;

/**
 * @author rah67
 *
 */
public class ConfusionMatrixDistribution implements ProbabilisticLabelErrorFunction.ConditionalCategoricalDistribution<Integer, Integer> {

  private final double[][] confusionMatrix;
  
  public ConfusionMatrixDistribution(double[][] confusionMatrix) {
    this.confusionMatrix = Preconditions.checkNotNull(confusionMatrix);
  }
  
  @Override
  public CategoricalDistribution<Integer> given(final Integer trueLabel) {
    Preconditions.checkArgument(0 <= trueLabel && trueLabel<confusionMatrix.length);
    return new CategoricalDistribution<Integer>() {

      @Override
      public Integer sample(RandomGenerator rnd) {
        double[] dist = confusionMatrix[trueLabel];
        return RandomGenerators.nextIntUnnormalizedProbs(rnd, dist);
      }
    };
  }
  
}