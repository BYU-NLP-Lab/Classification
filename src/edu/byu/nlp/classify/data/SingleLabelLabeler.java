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

import java.io.PrintWriter;
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import edu.byu.nlp.classify.Classifier;
import edu.byu.nlp.classify.ClassifierLearner;
import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.BasicDataset;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.AbstractRealMatrixPreservingVisitor;
import edu.byu.nlp.util.IncrementalAverager;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Pair;

/**
 * An implementation of {@code DatasetLabeler} that employs a standard ML learner and model.
 * Internally, it uses a {@code DatasetBuilder} to convert the data instances into classifier-
 * compatible instances, including making a decision about which label to use if there are
 * multiple labels.
 * 
 * @author rah67
 *
 */
public class SingleLabelLabeler implements DatasetLabeler {

  private final ClassifierLearner learner;
  private final DatasetBuilder datasetBuilder;
  private final int numAnnotators;
  private PrintWriter serializeOut;
  private boolean labelAllWithClassifier;

  public SingleLabelLabeler(ClassifierLearner learner, DatasetBuilder datasetBuilder,
      int numAnnotators) {
    this(learner, datasetBuilder, numAnnotators, null);
  }

  public SingleLabelLabeler(ClassifierLearner learner, DatasetBuilder datasetBuilder,
      int numAnnotators, PrintWriter serializeOut) {
    this(learner, datasetBuilder, numAnnotators, serializeOut, false);
  }
  public SingleLabelLabeler(ClassifierLearner learner, DatasetBuilder datasetBuilder,
      int numAnnotators, PrintWriter serializeOut, boolean labelAllWithClassifier) {
    this.learner = learner;
    this.datasetBuilder = datasetBuilder;
    this.numAnnotators = numAnnotators;
    this.serializeOut=serializeOut;
    this.labelAllWithClassifier=labelAllWithClassifier;
  }

  /** {@inheritDoc} */
  @Override
  public Predictions label(
      Dataset trainingData, Dataset heldoutInstances) {
    // FIXME(pfelt): There are two major problems with the logic below.
    // First: The two components making up the classifier's training data are NOT mutually exclusive so each instance could potentially 
    // be added twice. Once with a trusted label and once with an annotation-built label. 
    // Second: The unannotated data is not mutually exclusive with the training data (because of the trusted labels), 
    // so the classifier could be being tested on some of its own training data.
    // I'm not fixing these immediately because it imitates previous behavior in statnlp, which I am trying to replicate.
    
    Collection<Prediction> annotatedPredictions = Lists.newArrayList();
    Pair<? extends Dataset, ? extends Dataset> annotatedSplit = Datasets.divideInstancesWithAnnotations(trainingData);
    Dataset setWithAnnotations = annotatedSplit.getFirst();
    Dataset setWithoutAnnotations = annotatedSplit.getSecond();
    
    // 1) trusted observed labels
    Dataset setWithObservedLabels = Datasets.divideInstancesWithObservedLabels(trainingData).getFirst();
    // 2) labels "built" from annotations (e.g., majority vote). 
    Dataset annotationBasedLabeledData = datasetBuilder.buildDataset(setWithAnnotations, annotatedPredictions);
    
    // train a classifier on a training set assembled from components 1) and 2) above
    Iterable<DatasetInstance> classifierTrainingSetInstances = Iterables.concat(annotationBasedLabeledData,setWithObservedLabels);
    Dataset classifierTrainingSet = new BasicDataset(classifierTrainingSetInstances, trainingData.getMeasurements(), Datasets.infoWithUpdatedCounts(classifierTrainingSetInstances, trainingData.getInfo()));
    Classifier classifier = learner.learnFrom(classifierTrainingSet);

    // Optionally, re-do even the annotated instances with the classifier rather than the dataset builder labels 
    if (labelAllWithClassifier){
      annotatedPredictions = classifierPredictions(annotationBasedLabeledData, classifier);
    }
    
    // use the classifier to predict over the unannotated portions
    Collection<Prediction> unannotatedPredictions = classifierPredictions(setWithoutAnnotations, classifier);
    Collection<Prediction> heldoutPredictions = null;
    if (heldoutInstances!=null){
      heldoutPredictions = classifierPredictions(heldoutInstances, classifier);
    }
    serializePredictions(annotatedPredictions,unannotatedPredictions, serializeOut);
    double[] annotatorAccuracies = annotatorAccuracy(annotatedPredictions);
    double[][][] annotatorConfusionMatrices = annotatorConfusions(annotatedPredictions, trainingData.getInfo().getNumClasses());
    return new Predictions(annotatedPredictions, unannotatedPredictions,
        heldoutPredictions, annotatorAccuracies, annotatorConfusionMatrices , -1, null, -1);
  }


  /**
   * @param labeledPredictions
   * @param unlabeledPredictions
   * @param serializeOut2
   */
  private static void serializePredictions(
      Collection<Prediction> labeledPredictions,
      Collection<Prediction> unlabeledPredictions,
      PrintWriter serializeOut) {
    if (serializeOut!=null){
      int all[] = new int[labeledPredictions.size()+unlabeledPredictions.size()];
      int index = 0;
      for (Prediction prediction: labeledPredictions){
        all[index++] = prediction.getPredictedLabel();
      }
      for (Prediction prediction: unlabeledPredictions){
        all[index++] = prediction.getPredictedLabel();
      }
      serializeOut.write(IntArrays.toString(all));
    }
  }

  private Collection<Prediction> classifierPredictions(
      Dataset unlabeled, Classifier classifier) {
    Collection<Prediction> unlabeledPredictions = Lists.newArrayList();
    // get the predictions on the unlabeled data
    for (DatasetInstance instance : unlabeled) {
      List<Integer> predicted = classifier.classifyNBest(-1, instance.asFeatureVector());
      unlabeledPredictions.add(new BasicPrediction(predicted, instance));
    }
    return unlabeledPredictions;
  }

  private double[] annotatorAccuracy(Collection<Prediction> labeledPredictions) {
    // The number of times they disagree with the singly-labeled dataset 
    // (e.g., majority if DatasetBuilder had a MajorityVoteChooser).
    final IncrementalAverager<Long> avg = new IncrementalAverager<Long>();
    for (final Prediction p : labeledPredictions) {
      p.getInstance().getAnnotations().getLabelAnnotations().walkInOptimizedOrder(
          new AbstractRealMatrixPreservingVisitor() {
        @Override
        public void visit(int annotator, int annval, double value) {
          for (int i=0; i<value; i++){
            avg.addValue((long)annotator, annval==p.getPredictedLabel()? 1: 0);
          }
        }
      });
    }

    double[] annotatorAccuracies = new double[numAnnotators];
    for (int i = 0; i < annotatorAccuracies.length; i++) {
      annotatorAccuracies[i] = avg.average(new Long(i));
    }
    return annotatorAccuracies;
  }

  private double[][][] annotatorConfusions(
      Collection<Prediction> labeledPredictions, int numLabels) {
    // Calculate a matrix based on the disagreements with the singly-labeled dataset 
    // (e.g., majority if DatasetBuilder had a MajorityVoteChooser).
    final double[][][] confusionMatrices = new double[numAnnotators][numLabels][numLabels];
    if (numAnnotators>0 && numLabels>0){
      for (final Prediction p : labeledPredictions) {
        p.getInstance().getAnnotations().getLabelAnnotations().walkInOptimizedOrder(
            new AbstractRealMatrixPreservingVisitor() {
          @Override
          public void visit(int annotator, int annval, double value) {
            for (int i=0; i<value; i++){
              confusionMatrices[annotator][p.getPredictedLabel()][annval] += value;
            }
          }
        });
      }
    }

    return confusionMatrices;
  }
  
}