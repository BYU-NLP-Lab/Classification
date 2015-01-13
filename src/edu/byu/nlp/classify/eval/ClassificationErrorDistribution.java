/**
 * Copyright 2012 Brigham Young University
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

import com.google.common.annotations.VisibleForTesting;

import edu.byu.nlp.classify.eval.ProbabilisticLabelErrorFunction.CategoricalDistribution;
import edu.byu.nlp.classify.eval.ProbabilisticLabelErrorFunction.ConditionalCategoricalDistribution;




/**
 * NOTE: except for the static factory method, this class is more generic than the name implies.
 * @author rah67
 *
 */
public class ClassificationErrorDistribution implements ConditionalCategoricalDistribution<Integer,Integer> {

	private static class DistributionParameterizedByAccuracy implements CategoricalDistribution<Integer> {
		
		private final int label;
		private final double accuracy;
		private final int numLabels;
		
		public DistributionParameterizedByAccuracy(int thisLabel, int numLabels, double accuracy) {
			this.label = thisLabel;
			this.accuracy = accuracy;
			this.numLabels = numLabels;
		}

		/** {@inheritDoc} */
		@Override
		public Integer sample(RandomGenerator rnd) {
			double u = rnd.nextDouble();
			if (u < accuracy) {
				return label;
			}
			// To sample from the rest of the labels, we use the uniform deviate to sample an integer between
			// 0 and numLabels - 1 (exclusive). Note, however, that these indices have to be shifted to account
			// for the fact that this.label is no longer an option.
			int choice = (int) ((u - accuracy) * (numLabels - 1));
			if (choice >= label) {
				++choice;
			}
			assert choice < numLabels;
			return choice;
		}
		
	}
	
	private final CategoricalDistribution<Integer>[] distributions;
	
	@VisibleForTesting
	ClassificationErrorDistribution(CategoricalDistribution<Integer>[] distributions) {
		this.distributions = distributions;
	}
	
	/** {@inheritDoc} */
	@Override
	public CategoricalDistribution<Integer> given(Integer condition) {
		return distributions[condition];
	}
	
	/**
	 * Generates a conditional distribution where p(correct) is the supplied accuracy and the various errors are
	 * uniformly distributed across the other options. 
	 */
	public static ClassificationErrorDistribution usingAccuracy(int numLabels, double accuracy) {
		@SuppressWarnings("unchecked")
		CategoricalDistribution<Integer>[] distributions = new CategoricalDistribution[numLabels];
		for (int label = 0; label < numLabels; label++) {
			distributions[label] = new DistributionParameterizedByAccuracy(label, numLabels, accuracy);
		}
		return new ClassificationErrorDistribution(distributions);
	}
	
}
