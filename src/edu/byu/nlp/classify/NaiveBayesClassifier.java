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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;

import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.math.Math2;
import edu.byu.nlp.stats.CategoricalDistribution;
import edu.byu.nlp.stats.DoubleArrayCategoricalDistribution;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Matrices;

/**
 * @author rah67
 *
 */
public class NaiveBayesClassifier extends LinearClassifier implements ProbabilisticClassifier {

    public static NaiveBayesClassifier newClassifier(double[] logPOfY, double[] logPOfXGivenY, boolean copy) {
        Preconditions.checkNotNull(logPOfY);
        Preconditions.checkNotNull(logPOfXGivenY);
        Preconditions.checkArgument(logPOfXGivenY.length % logPOfY.length == 0);
        
        if (copy) {
            logPOfY = logPOfY.clone();
            logPOfXGivenY = logPOfXGivenY.clone();
        }
        return new NaiveBayesClassifier(logPOfY, logPOfXGivenY);
    }
    
    public static NaiveBayesClassifier newClassifier(double[] logPOfY, double[] logPOfXGivenY, boolean copy,
                                                     double tolerance) {
        Preconditions.checkNotNull(logPOfY);
        Preconditions.checkNotNull(logPOfXGivenY);
        Preconditions.checkArgument(logPOfXGivenY.length % logPOfY.length == 0);

        if (copy) {
            logPOfY = logPOfY.clone();
            logPOfXGivenY = logPOfXGivenY.clone();
        }
        
        double logSum = DoubleArrays.logSum(logPOfY);
        if (!Math2.doubleEquals(logSum, 0.0, tolerance)) {
            throw new IllegalArgumentException(
                    "Improper distribution: log(sum) = " + logSum + "; sum = " + Math.exp(logSum));
        }
        
        for (int k = 0; k < logPOfY.length; k++) {
            logSum = Matrices.logSumRowInColumnMajorMatrix(logPOfXGivenY, logPOfY.length, k);
            if (!Math2.doubleEquals(logSum, 0.0, tolerance)) {
                throw new IllegalArgumentException(
                        "Improper distribution for Y = " + k + ": log(sum) = " + logSum + 
                        "; sum = " + Math.exp(logSum));
            }
        }
        
        return new NaiveBayesClassifier(logPOfY, logPOfXGivenY);
    }
    

    @VisibleForTesting NaiveBayesClassifier(double[] logPOfY, double[] logPOfXGivenY) {
        super(logPOfY, logPOfXGivenY);
    }
    
    /** {@inheritDoc} */
    @Override
    public CategoricalDistribution given(SparseFeatureVector condition) {
        // The scores of a linear model are the dot product between the input vector and weights for each class.
        // Since each weight w_yf is log p(x_f | y), p(y|x) \propto x \cdot w_y.
        double[] scores = scoresFor(condition);
        DoubleArrays.logNormalizeToSelf(scores);
        return DoubleArrayCategoricalDistribution.newDistributionFromLogProbs(scores, true);
    }

}
