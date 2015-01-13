/**
 * Copyright 2013 Brigham Young University
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

import java.util.Collections;
import java.util.List;

import com.google.common.base.Preconditions;

import edu.byu.nlp.data.types.DatasetInstance;

public class BasicPrediction implements Prediction {

	private final List<Integer> predictedLabels;
	private final DatasetInstance instance;

  
	public BasicPrediction(Integer predictedLabel, DatasetInstance instance) {
		this(Collections.singletonList(predictedLabel), instance);
	}
	
  public BasicPrediction(List<Integer> predictedLabels, DatasetInstance instance) {
    Preconditions.checkNotNull(predictedLabels);
    Preconditions.checkNotNull(instance);
    Preconditions.checkArgument(predictedLabels.size()>0);
    this.predictedLabels=predictedLabels;
    this.instance = instance;
  }

	/** {@inheritDoc} */
	@Override
	public Integer getPredictedLabel() {
		return predictedLabels.get(0);
	}

	/** {@inheritDoc} */
	@Override
	public DatasetInstance getInstance() {
		return instance;
	}

  /** {@inheritDoc} */
  @Override
  public String toString() {
      return "BasicPrediction [predictedLabel=" + getPredictedLabel() + ", instance=" + instance + "]";
  }

  /** {@inheritDoc} */
  @Override
  public List<Integer> getPredictedLabels() {
    return predictedLabels;
  }
}