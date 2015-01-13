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

import com.google.common.collect.Iterables;

public class Predictions {
  private final Iterable<? extends Prediction> labeledPredictions;
  private final Iterable<? extends Prediction> unlabeledPredictions;
  private final Iterable<? extends Prediction> heldoutPredictions;
  private final double[] annotatorAccuracies;
  private final double[][][] annotatorConfusionMatrices;
  private final double machineAccuracy;
  private final double[][] machineConfusionMatrix; 
  private final double logJoint;

  public Predictions(Iterable<? extends Prediction> labeledPredictions,
      Iterable<? extends Prediction> unlabeledPredictions,
          Iterable<? extends Prediction> heldoutPredictions,
              double[] annotatorAccuracies, double[][][] annotatorConfusionMatrices, 
              double machineAccuracy, double[][] machineConfusionMatrix, double logJoint) {
    this.labeledPredictions = labeledPredictions;
    this.unlabeledPredictions = unlabeledPredictions;
    this.heldoutPredictions = heldoutPredictions;
    this.annotatorAccuracies = annotatorAccuracies;
    this.annotatorConfusionMatrices=annotatorConfusionMatrices;
    this.machineAccuracy = machineAccuracy;
    this.machineConfusionMatrix=machineConfusionMatrix;
    this.logJoint = logJoint;
  }

  public Iterable<? extends Prediction> labeledPredictions() {
    return labeledPredictions;
  }

  public Iterable<? extends Prediction> unlabeledPredictions() {
    return unlabeledPredictions;
  }

  public Iterable<? extends Prediction> testPredictions() {
    return heldoutPredictions;
  }

  public Iterable<? extends Prediction> allPredictions() {
    return Iterables.concat(labeledPredictions, unlabeledPredictions, heldoutPredictions);
  }

  public double[] annotatorAccuracies() {
    return annotatorAccuracies;
  }
  
  public double[][][] annotatorConfusionMatrices(){
    return annotatorConfusionMatrices;
  }
  
  public double machineAccuracy() {
    return machineAccuracy;
  }
  
  public double[][] machineConfusionMatrix(){
    return machineConfusionMatrix;
  }
  
  public double logJoint(){
    return logJoint;
  }
}