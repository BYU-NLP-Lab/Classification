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

import java.util.List;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;

import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.util.DoubleArrays;

/**
 * @author rah67
 *
 */
public class LinearClassifier implements Classifier {

	private final double[] bias;
	// TODO(rah67): this should really be represented by a ColumnMajorMatrix class
	private final double[] weights;
	private final double[] scores;
	private final int numFeatures;
	
	public LinearClassifier(double[] bias, double... weights) {
		Preconditions.checkNotNull(bias);
		Preconditions.checkNotNull(weights);
		Preconditions.checkArgument(weights.length % bias.length == 0,
				"There is a mismatch in the number of classes (%s) and the number of weights (%s)",
				Integer.toString(bias.length), Integer.toString(weights.length));
		
		this.numFeatures = weights.length / bias.length;
		this.bias = bias;
		this.weights = weights;
		this.scores = new double[bias.length];
	}
	
	/** {@inheritDoc} */
	@Override
	public int classify(SparseFeatureVector s) {
		return DoubleArrays.argMax(scoresFor(s));
	}
	
	/**
	 * The returned array is owned by this class and may be changed at any time. Therefore a copy should be
	 * made before any other operations are performed with the classifier.
	 */
	protected double[] scoresFor(SparseFeatureVector s) {
	  Preconditions.checkArgument(s.length() <= numFeatures, "The input vector is longer than the number of features");
    System.arraycopy(bias, 0, scores, 0, bias.length);
    // FIXME(rah67): be sure there that "unseen" features don't AIOOB
    s.preMultiplyAsColumnAndAddTo(weights, scores);
    return scores;
	}
	
	@VisibleForTesting double[] getBias() { return bias; }
	@VisibleForTesting double[] getWeights() { return weights; }

  /** {@inheritDoc} */
  @Override
  public List<Integer> classifyNBest(int n, SparseFeatureVector s) {
    return DoubleArrays.argMaxList(n,scoresFor(s));
  }
	
}
