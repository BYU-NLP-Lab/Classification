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

import com.google.common.base.Function;

/**
 * A LabelErrorFunction that, given a true label, samples from the distribution of "noisy" labels.
 * 
 * @author rah67
 *
 */
public class ProbabilisticLabelErrorFunction<L> implements Function<L, L> {

	// FIXME(rah67): reconcile this with the versions in the stats package
	public static interface CategoricalDistribution<E> {
//		double logProbabilityOf(E event);
//		E argMax();
		E sample(RandomGenerator rnd);
		// Sampler<E> sampler(RandomGenerator rnd);
	}
	
	// TODO(rhaertel): use ConditionalLogCategoricalDistribution
	public static interface ConditionalCategoricalDistribution<C, E> {
		CategoricalDistribution<E> given(C condition);
	}

	private final ConditionalCategoricalDistribution<L, L> dist;
	private final RandomGenerator rnd;
	
	public ProbabilisticLabelErrorFunction(ConditionalCategoricalDistribution<L, L> dist, RandomGenerator rnd) {
		this.dist = dist;
		this.rnd = rnd;
	}

	/** {@inheritDoc} */
	@Override
	public L apply(L trueLabel) {
		return dist.given(trueLabel).sample(rnd);
	}

}
