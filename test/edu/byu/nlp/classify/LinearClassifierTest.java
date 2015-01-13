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

import static org.fest.assertions.Assertions.assertThat;
import static org.fest.assertions.Delta.delta;

import java.util.Arrays;

import org.junit.Test;

import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.types.SparseFeatureVector.Entry;
import edu.byu.nlp.dataset.BasicSparseFeatureVector;
import edu.byu.nlp.util.DoubleArrays;

/**
 * @author rah67
 * @author plf1
 *
 */
public class LinearClassifierTest {

	private double[] expectedScores(double[] bias, double[] weights, SparseFeatureVector s) {
		double[] scores = bias.clone();
		for (Entry entry : s.sparseEntries()) {
			for (int k = 0; k < scores.length; k++) {
				scores[k] += weights[entry.getIndex() * bias.length + k] * entry.getValue();
			}
		}
//		System.out.println(Arrays.toString(scores));
		return scores;
	}
	
	/**
	 * Test method for {@link edu.byu.nlp.classify.LinearClassifier#classify(edu.byu.nlp.data.SparseFeatureVector)}.
	 */
	@Test
	public void testScores() {
		// Column-major matrix (3 classes x 4 features)
		double[] bias = new double[] { -1.1, 0.1, 2.1 };
		double[] weights = new double[] {
				-1.0,  0.0,   1.0,
				10.0,  1.0,   0.1,
				 0.1, -10.0, -0.1,
				 1.0,  1.0,  10.0,
				 1.0,  1.0,   1.0};
		LinearClassifier classifier = new LinearClassifier(bias, weights);
		
		SparseFeatureVector v1 = new BasicSparseFeatureVector(
				new int[] {0, 4}, new double[] {1.23, 3.14});
		assertThat(classifier.scoresFor(v1)).isEqualTo(expectedScores(bias, weights, v1), delta(1e-8));

		SparseFeatureVector v2 = new BasicSparseFeatureVector(
				new int[] {0, 1, 2, 3, 4}, new double[] {-1.23, 2.45, -3.21, 4.72, 0.01});
		assertThat(classifier.scoresFor(v2)).isEqualTo(expectedScores(bias, weights, v2), delta(1e-8));

		SparseFeatureVector v3 = new BasicSparseFeatureVector(
				new int[] {0, 1, 2, 3, 4}, new double[] {-1.23, 2.45, 4.72, -3.21, 0.01});
		assertThat(classifier.scoresFor(v3)).isEqualTo(expectedScores(bias, weights, v3), delta(1e-8));
	}

    @Test
    public void testClassify() {
        // Column-major matrix (3 classes x 4 features)
        double[] bias = new double[] { -1.1, 0.1, 2.1 };
        double[] weights = new double[] {
                -1.0,  0.0,   1.0,
                10.0,  1.0,   0.1,
                 0.1, -10.0, -0.1,
                 1.0,  1.0,  10.0,
                 1.0,  1.0,   1.0};
        LinearClassifier classifier = new LinearClassifier(bias, weights);
        
        SparseFeatureVector v1 = new BasicSparseFeatureVector(
                new int[] {0, 4}, new double[] {1.23, 3.14});
        int expectedLabel = DoubleArrays.argMax(expectedScores(bias, weights, v1));
        assertThat(classifier.classify(v1)).isEqualTo(expectedLabel);
        
        SparseFeatureVector v2 = new BasicSparseFeatureVector(
                new int[] {0, 1, 2, 3, 4}, new double[] {-1.23, 2.45, -3.21, 4.72, 0.01});
        expectedLabel = DoubleArrays.argMax(expectedScores(bias, weights, v2));
        assertThat(classifier.classify(v2)).isEqualTo(expectedLabel);

        SparseFeatureVector v3 = new BasicSparseFeatureVector(
                new int[] {0, 1, 2, 3, 4}, new double[] {-1.23, 2.45, 4.72, -3.21, 0.01});
        expectedLabel = DoubleArrays.argMax(expectedScores(bias, weights, v3));
        assertThat(classifier.classify(v3)).isEqualTo(expectedLabel);
    }
}
