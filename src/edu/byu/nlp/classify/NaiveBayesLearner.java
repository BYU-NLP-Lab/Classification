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
package edu.byu.nlp.classify;

import java.util.Arrays;

import com.google.common.base.Preconditions;

import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.util.ColumnMajorMatrices;
import edu.byu.nlp.util.DoubleArrays;


/**
 * Add one smoothing.
 * @author rah67
 *
 */
public class NaiveBayesLearner implements ClassifierLearner {

	/** {@inheritDoc} */
	@Override
	public NaiveBayesClassifier learnFrom(Dataset data) {
		Preconditions.checkNotNull(data);
		Preconditions.checkArgument(data.getInfo().getNumClasses() > 0, "Dataset must have at least one class");
		Preconditions.checkArgument(data.getInfo().getNumFeatures() >= 0, "Dataset must have zero or more features");
		
		Dataset labeledData = Datasets.divideInstancesWithObservedLabels(data).getFirst();
		
		double[] weights = new double[data.getInfo().getNumClasses() * data.getInfo().getNumFeatures()];
		double[] biases = new double[data.getInfo().getNumClasses()];

		// Add-one smoothing.
		Arrays.fill(weights, 1.0);
		Arrays.fill(biases, 1.0);
		
		for (DatasetInstance instance : labeledData) {
			++biases[instance.getObservedLabel()];
			instance.asFeatureVector().addToRow(weights, instance.getObservedLabel(), data.getInfo().getNumClasses());
		}

		// Compute log p(c)
		DoubleArrays.logToSelf(biases);
		DoubleArrays.logNormalizeToSelf(biases);
		
		// Compute log p(f|c)
    ColumnMajorMatrices.normalizeRows(weights, biases.length);
		DoubleArrays.logToSelf(weights);
		
		return new NaiveBayesClassifier(biases, weights);
	}

}
