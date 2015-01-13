/**
 * Copyright 2011 Brigham Young University
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

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.byu.nlp.dataset.BasicSparseFeatureVector;
import edu.byu.nlp.util.asserts.MoreAsserts;

/**
 * Unit tests for {@link LinearClassifier}.
 * 
 * @author rah67
 *
 */
public class LinearModelTest {

	/**
	 * Test method for {@link edu.byu.nlp.classify.LinearClassifier#scores(double[][], double[], int)}.
	 */
	@Test
	public void testScores() {
		double[] weights = new double[]{-0.5, 0.5, 1.0, -3.5}; 
		final LinearClassifier model = new LinearClassifier(new double[]{3,1}, weights);

		// Null input
		try {
			model.scoresFor(null);
		} catch (NullPointerException expected) {}
		
		// Too few features is ok (model assumes that sparse entries are 0)
    model.scoresFor(BasicSparseFeatureVector.fromDenseFeatureVector(new double[]{0})); 
		
		// Too many features
    MoreAsserts.assertFails(new Runnable() {
      @Override
      public void run() {
        model.scoresFor(BasicSparseFeatureVector.fromDenseFeatureVector(new double[]{0,0,0,0,0})); 
      }
    }, IllegalArgumentException.class);
		
		// well-formed input
		double[] scores = model.scoresFor(BasicSparseFeatureVector.fromDenseFeatureVector(new double[]{1.5,-1.5}));
		
		assertEquals((-0.5*1.5) + (1.0*-1.5) + 3, scores[0], 1e-14);
		assertEquals((0.5*1.5) + (-3.5*-1.5) + 1, scores[1], 1e-14);
	}

}
