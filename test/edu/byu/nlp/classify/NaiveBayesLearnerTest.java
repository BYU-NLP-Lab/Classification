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

import static edu.byu.nlp.util.DoubleArrays.log;

import java.util.List;

import org.fest.assertions.Assertions;
import org.fest.assertions.Delta;
import org.junit.Test;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import edu.byu.nlp.data.streams.IndexerCalculator;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.BasicDataset;
import edu.byu.nlp.dataset.BasicDatasetInstance;
import edu.byu.nlp.dataset.BasicSparseFeatureVector;
import edu.byu.nlp.math.Math2;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Indexer;

/**
 * @author rah67
 *
 */
public class NaiveBayesLearnerTest {

	/**
	 * Test method for {@link edu.byu.nlp.classify.NaiveBayesLearner#learnFrom(edu.byu.nlp.al.classify2.Dataset)}.
	 */
	@Test
	public void testLearnFrom() {

	  // Create Indexers
	  Indexer<String> annotatorIdIndexer = new Indexer<String>();
	  // 0 annotators
	  // 5 instances
    Indexer<String> instanceIdIndexer = new Indexer<String>();
    for (long i=0; i<5; i++){
      instanceIdIndexer.add(""+i);
    }
    // 4 features
    Indexer<String> featureIndexer = new Indexer<String>();
    for (int f=0; f<4; f++){
      featureIndexer.add(""+f);
    }
    // 3 labels
    Indexer<String> labelIndexer = new Indexer<String>();
    for (int l=0; l<3; l++){
      labelIndexer.add(""+l);
    }
    
    // Create Dataset
    List<DatasetInstance> instances = Lists.newArrayList();
    int instanceId = 0;
    instances.add(new BasicDatasetInstance(new BasicSparseFeatureVector(new int[]{0, 2, 3}, new double[]{1., 2., 3.}),  2, instanceId++, "0", labelIndexer));
    instances.add(new BasicDatasetInstance(new BasicSparseFeatureVector(new int[]{3, 2, 1}, new double[]{4., 5., 6.}),  0, instanceId++, "0", labelIndexer));
    instances.add(new BasicDatasetInstance(new BasicSparseFeatureVector(new int[]{}, new double[]{}),                   1, instanceId++, "0", labelIndexer));
    instances.add(new BasicDatasetInstance(new BasicSparseFeatureVector(new int[]{1}, new double[]{7.}),                1, instanceId++, "0", labelIndexer));
    instances.add(new BasicDatasetInstance(new BasicSparseFeatureVector(new int[]{1}, new double[]{8.}),                0, instanceId++, "0", labelIndexer));
    Dataset dataset = new BasicDataset("", instances, Sets.newHashSet(), new IndexerCalculator<>(featureIndexer, labelIndexer, instanceIdIndexer, annotatorIdIndexer));
	  
		//
		// Compute the weights that we "expect"
		//
		
		// The number of times each class occurs (and add-one smoothing!)
		double[] logPOfC = new double[] { 1 + 2, 1 + 2, 1 + 1 };
		DoubleArrays.logToSelf(logPOfC);
		DoubleArrays.logNormalizeToSelf(logPOfC);
		
		// The sum over instances for each class is:
		//   Class 0: [0., 14., 5., 4.]
		//   Class 1: [0.,  7., 0., 0.]
		//   Class 2: [1.,  0., 2., 3.]
		// (Don't forget add-one smoothing!
		double[] featureClassSums = new double[] {
				1 +  0., 1 + 0., 1 + 1., 
				1 + 14., 1 + 7., 1 + 0.,
				1 +  5., 1 + 0., 1 + 2.,
				1 +  4., 1 + 0., 1 + 3. };
		double[] norm = new double[] {
				featureClassSums[0] + featureClassSums[3] + featureClassSums[6] + featureClassSums[9],
				featureClassSums[1] + featureClassSums[4] + featureClassSums[7] + featureClassSums[10],
				featureClassSums[2] + featureClassSums[5] + featureClassSums[8] + featureClassSums[11] };
		double[] pOfFeatureGivenClass = new double[] {
			featureClassSums[0] / norm[0], featureClassSums[1] / norm[1], featureClassSums[2] / norm[2],
			featureClassSums[3] / norm[0], featureClassSums[4] / norm[1], featureClassSums[5] / norm[2],
			featureClassSums[6] / norm[0], featureClassSums[7] / norm[1], featureClassSums[8] / norm[2],
			featureClassSums[9] / norm[0], featureClassSums[10] / norm[1], featureClassSums[11] / norm[2] };
		assert isNormalized(pOfFeatureGivenClass, 3, 4);
		
		//
		// Train the classifier to get the "actual" weights 
		//
		LinearClassifier classifier = new NaiveBayesLearner().learnFrom(dataset);
		
		Assertions.assertThat(classifier.getBias()).isEqualTo(logPOfC, Delta.delta(1e-8));
		Assertions.assertThat(classifier.getWeights()).isEqualTo(log(pOfFeatureGivenClass), Delta.delta(1e-8));
	}

	private boolean isNormalized(double[] pOfFeatureGivenClass, int numLabels, int numFeatures) {
		for (int label = 0; label < numLabels; label++) {
			double sum = 0.0;
			for (int f = 0; f < numFeatures; f++) {
				sum += pOfFeatureGivenClass[f * numLabels + label];
			}
			if (!Math2.doubleEquals(sum, 1.0, 1e-8)) {
				return false;
			}
		}
		return true;
	}

}
