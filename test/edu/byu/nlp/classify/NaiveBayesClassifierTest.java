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

import static java.lang.Math.log;
import static org.fest.assertions.Assertions.assertThat;

import org.junit.Test;

import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.dataset.BasicSparseFeatureVector;
import edu.byu.nlp.util.DoubleArrays;

/**
 * @author rah67
 *
 */
public class NaiveBayesClassifierTest {

    /**
     * Test method for {@link edu.byu.nlp.classify.NaiveBayesClassifier#given(edu.byu.nlp.data.SparseFeatureVector)}.
     */
    @Test
    public void testGiven() {
        double[] logPOfY = new double[] { log(0.1), log(0.45), log(0.45) };
        // Column-major (features are columns)
        double[] logPOfXGivenY = new double[] { log(0.8), log(0.2), log(0.1),
                                                log(0.1), log(0.7), log(0.3),
                                                log(0.1), log(0.1), log(0.6) };
        NaiveBayesClassifier classifier = NaiveBayesClassifier.newClassifier(logPOfY, logPOfXGivenY, false, 1e-14);
        
        double[] values = new double[]{ 1.1, 2.2, 3.3 };
        SparseFeatureVector v1 = new BasicSparseFeatureVector(new int[]{0, 1, 2}, values);
        
        double expectedLogPOf0 = logPOfY[0] + values[0] * logPOfXGivenY[0] + 
                                              values[1] * logPOfXGivenY[3] +
                                              values[2] * logPOfXGivenY[6]; 
        
        double expectedLogPOf1 = logPOfY[1] + values[0] * logPOfXGivenY[1] + 
                                              values[1] * logPOfXGivenY[4] +
                                              values[2] * logPOfXGivenY[7]; 
        
        double expectedLogPOf2 = logPOfY[2] + values[0] * logPOfXGivenY[2] + 
                                              values[1] * logPOfXGivenY[5] +
                                              values[2] * logPOfXGivenY[8];
        
        // Normalize
        double logSum = DoubleArrays.logSum(expectedLogPOf0, expectedLogPOf1, expectedLogPOf2);
        expectedLogPOf0 -= logSum;
        expectedLogPOf1 -= logSum;
        expectedLogPOf2 -= logSum;
        
        assertThat(classifier.given(v1).logProbabilityOf(0)).isEqualTo(expectedLogPOf0);
        assertThat(classifier.given(v1).logProbabilityOf(1)).isEqualTo(expectedLogPOf1);
        assertThat(classifier.given(v1).logProbabilityOf(2)).isEqualTo(expectedLogPOf2);
    }

}
