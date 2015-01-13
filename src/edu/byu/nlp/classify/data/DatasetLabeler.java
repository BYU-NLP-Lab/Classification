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
package edu.byu.nlp.classify.data;

import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.data.types.Dataset;

/**
 * An interface for producing predictions on an entire dataset. Said dataset may have
 * multiple annotations per instance. Algorithms that directly incorporate this information
 * may implement this interface directly. {@code SingleLabelLabeler} allows one to use
 * standard machine learning algorithms without modification.
 * 
 * @author rah67
 */
public interface DatasetLabeler {
  /**
   * Performs the prediction using the provided instances. The instances may included
   * labeled and / or unlabeled data. The labeled data may be used to train a model to
   * produce predictions on the remaining instances.  
   */
	Predictions label(Dataset trainingInstances, Dataset heldoutInstances);
}