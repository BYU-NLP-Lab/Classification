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

import edu.byu.nlp.util.Indexer;

public class ConfusionMatrixComputer {
	
  private final Indexer<String> labels;

  public ConfusionMatrixComputer(Indexer<String> labels){
    this.labels=labels;
  }
  
	public ConfusionMatrix compute(Iterable<? extends Prediction> predictions) {
    ConfusionMatrix matrix = new ConfusionMatrix(labels.size(), labels.size(), labels);
    for (Prediction prediction : predictions) {
      int truth = prediction.getInstance().getLabel();
      int guess = prediction.getPredictedLabel();
      matrix.addToEntry(truth, guess, 1);
    }
    return matrix;
	}
	
	public String csvHeader() {
	    return "labeled_correct, labeled_total, labeled_acc, unlabeled_correct, unlabeled_total, " +
	           "unlabeled_acc, overall_correct, overall_total, overall_acc, heldout_correct, " +
	           "heldout_total, heldout_acc";
	}
}