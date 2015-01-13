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

import java.util.List;

public class AccuracyComputer {
  private int topn;

  public AccuracyComputer(){
    this(1);
  }
	public AccuracyComputer(int topn){
	  this.topn=topn;
	}
	private Accuracy computeFor(Iterable<? extends Prediction> predictions, Integer nullLabel) {
		int correct = 0;
		int total = 0;
		for (Prediction prediction : predictions) {
		  if (!prediction.getInstance().hasLabel()){
		    // ignore items with the null label
		    // we truly don't have the labels for these--we aren't just hiding them
		    continue; 
		  }
		  else if (prediction.getPredictedLabel()==null){
        // some data instances might have null predictions because they were removed 
        // by the model as not-applicable. For example, the itemresp model removes 
        // from consideration all instances that have no annotations. It has no way of 
        // inferring good labels for these. Rather than guessing a random label, it 
        // punts and returns null. The model is not penalized for these punts.
        // ignore
		    continue;
		  }
		  else{
		    // we are right if any of the top n predictions are correct
		    int maxguess = Math.min(topn,prediction.getPredictedLabels().size());
		    Integer gold = prediction.getInstance().getLabel();
		    List<Integer> guesses = prediction.getPredictedLabels().subList(0, maxguess);
		    if (guesses.contains(gold)){
		      ++correct;
		    }
  			++total;
		  }
		}
		return new Accuracy(correct, total);
	}
	
	public OverallAccuracy compute(Predictions predictions, Integer nullLabel) {
		Accuracy labeledAccuracy = computeFor(predictions.labeledPredictions(), nullLabel);
		Accuracy unlabeledAccuracy = computeFor(predictions.unlabeledPredictions(), nullLabel);
		Accuracy heldoutAccuracy = computeFor(predictions.testPredictions(), nullLabel);
		return new OverallAccuracy(labeledAccuracy, unlabeledAccuracy, heldoutAccuracy);
	}
	
	public String csvHeader() {
	  String prefix = (topn==1)? "": "top"+topn+"_";
	    return prefix+"labeled_correct, "+prefix+"labeled_total, "+prefix+"labeled_acc, "+
	      prefix+"unlabeled_correct, "+prefix+"unlabeled_total, "+prefix+"unlabeled_acc, "+
	      prefix+"overall_correct, "+prefix+"overall_total, "+prefix+"overall_acc, "+
	      prefix+"heldout_correct, "+prefix+"heldout_total, "+prefix+"heldout_acc";
	}
}