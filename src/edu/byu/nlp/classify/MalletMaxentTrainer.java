/**
 * Copyright 2014 Brigham Young University
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

import cc.mallet.classify.MaxEnt;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Label;
import cc.mallet.types.LabelAlphabet;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.types.SparseFeatureVector.EntryVisitor;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Enumeration;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Iterables2;

public class MalletMaxentTrainer{

  private cc.mallet.types.Instance[] instances;
  private Alphabet dataAlphabet;
  private LabelAlphabet targetAlphabet;
  private List<DatasetInstance> externalInstances;

  private MalletMaxentTrainer(){}
  
  /**
   * The trainer takes care of converting the dataset into types that 
   * mallet can work with. Labels are not converted here, but 
   * are passed in separately during training.
   */
  public static MalletMaxentTrainer build(Dataset data){

    final MalletMaxentTrainer trainer = new MalletMaxentTrainer();
    
    trainer.externalInstances = Lists.newArrayListWithCapacity(data.getInfo().getNumDocuments());
    trainer.instances = new cc.mallet.types.Instance[data.getInfo().getNumDocuments()];
    trainer.dataAlphabet = new Alphabet();
    trainer.dataAlphabet.startGrowth();
    trainer.targetAlphabet = new LabelAlphabet();
    trainer.targetAlphabet.startGrowth();
    
    // create identity mallet alphabets (so that our indices correspond exactly to theirs)
    for (int f=0; f<data.getInfo().getNumFeatures(); f++){
      trainer.dataAlphabet.lookupIndex(f,true);
    }
    trainer.dataAlphabet.stopGrowth();
    for (int l=0; l<data.getInfo().getNumClasses(); l++){
      trainer.targetAlphabet.lookupLabel(l,true);
    }
    trainer.targetAlphabet.stopGrowth();
    
    // alphabet sanity check #1 (make sure mallet alphabets return identity mappings
    Preconditions.checkState(data.getInfo().getNumFeatures()==trainer.dataAlphabet.size());
    Preconditions.checkState(data.getInfo().getNumClasses()==trainer.targetAlphabet.size());
    for (int f=0; f<trainer.dataAlphabet.size(); f++){
      Preconditions.checkState(trainer.dataAlphabet.lookupIndex(f)==f);
      Preconditions.checkState(trainer.dataAlphabet.lookupObject(f).equals(new Integer(f)));
    }
    for (int f=0; f<trainer.targetAlphabet.size(); f++){
      Preconditions.checkState(trainer.targetAlphabet.lookupIndex(f)==f);
      Preconditions.checkState(trainer.targetAlphabet.lookupLabel(f).getIndex()==f);
      Preconditions.checkState(trainer.targetAlphabet.lookupObject(f).equals(new Integer(f)));
    }
    
    // alphabet sanity check #2 (make sure every instance in the data has valid mallet data and label alphabet entries) 
    // this would only fail if our indexers were not working properly (there was a gap somewhere among 
    // the existing indices)
    for (DatasetInstance inst: data){
      // visit the data (to make sure all features and labels were added correctly)
      inst.asFeatureVector().visitSparseEntries(new EntryVisitor() {
        @Override
        public void visitEntry(int index, double value) {
          Preconditions.checkState(trainer.dataAlphabet.lookupIndex(index,false)>=0);
        }
      });
      if (inst.hasLabel()){ // ignore null label but use hidden labels to help ensure good alphabet coverage (no cheating here)
        Preconditions.checkState(trainer.targetAlphabet.lookupIndex(inst.getLabel())>=0);
      }
    }
    
    // convert each dataset instance to a mallet instance 
    for (Enumeration<DatasetInstance> item: Iterables2.enumerate(data)){
      // convert feature vector
      trainer.instances[item.getIndex()] = convert(trainer.dataAlphabet, item.getElement());
      // remember the original instance
      trainer.externalInstances.add(item.getElement());
    }

    return trainer;
  }
  
  /**
   * Train a log-linear model using the given soft labels (must match the 
   * dataset this trainer was build on).
   */
  public MaxEnt maxDataModel(double[][] softlabels, MaxEnt previousModel){
    // create a training set by adding each instance K times, each weighted by softlabels
    InstanceList trainingSet = new InstanceList(dataAlphabet, targetAlphabet);
    for (int i=0; i<instances.length; i++){
      for (int k=0; k<targetAlphabet.size(); k++){
        if (!Double.isNaN(softlabels[i][k])){ // ignore nans (instances with no annotations)
          // give this instance label k with weight softlabels[k]
          cc.mallet.types.Instance inst = instances[i].shallowCopy();
          inst.setTarget(targetAlphabet.lookupLabel(k));
          trainingSet.add(inst, softlabels[i][k]);
        }
      }
    }
    // train
    return new MaxEntTrainer(previousModel).train(trainingSet);
  }
  
  /**
   * Convert a single DatasetInstance to a mallet instance with no label
   */
  public static cc.mallet.types.Instance convert(final Alphabet dataAlphabet, DatasetInstance inst){
    return convert(dataAlphabet, inst.asFeatureVector(), inst.getInfo().getRawSource());
  }

  public static cc.mallet.types.Instance convert(final Alphabet dataAlphabet, SparseFeatureVector features, String source){
    final List<Integer> featureIndices = Lists.newArrayList();
    final List<Double> featureValues = Lists.newArrayList();
    features.visitSparseEntries(new EntryVisitor() {
      @Override
      public void visitEntry(int index, double value) {
        int featureIndex = dataAlphabet.lookupIndex(index);
        if (featureIndex>=0){ // ignore unknown features (for generalization)
          featureIndices.add(dataAlphabet.lookupIndex(index));
          featureValues.add(value);
        }
      }
    });
    
    // add to trainingData
    FeatureVector malletFV = new FeatureVector(dataAlphabet, IntArrays.fromList(featureIndices), DoubleArrays.fromList(featureValues));
    String name = source;
    Label target = null; // no label for now
    
    // only convert instances with non-null labels
    return new cc.mallet.types.Instance(malletFV, target, name, source);
  }
  
  /**
   * Get the weights w from the underlying log-linear model as a double[class][feature].
   * The final entry of each row (i.e., w[class][numFeatures]) is the class bias weight.
   */
  public double[][] maxWeights(MaxEnt maxent, int numClasses, int numFeatures){
    Preconditions.checkState(maxent.getNumParameters()==numClasses*numFeatures+numClasses);
    Preconditions.checkState(maxent.getDefaultFeatureIndex()==numFeatures);
    double[][] maxLambda = new double[numClasses][numFeatures+1]; // +1 accounts for class bias term
    double[] params = maxent.getParameters();
    for (int k=0; k<numClasses; k++){
      int srcPos = k*(numFeatures+1);
      int destPos = 0;
      int length = numFeatures+1;
      System.arraycopy(params, srcPos, maxLambda[k], destPos, length);
    }
    return maxLambda;
  }
} // end of trainer