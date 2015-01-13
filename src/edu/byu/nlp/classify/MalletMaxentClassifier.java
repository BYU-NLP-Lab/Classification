/**
 * Copyright 2015 Brigham Young University
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
import cc.mallet.types.Instance;

import com.google.common.collect.Lists;

import edu.byu.nlp.data.types.SparseFeatureVector;

/**
 * @author pfelt
 *
 */
public class MalletMaxentClassifier implements Classifier{

  private final MaxEnt maxent;

  public MalletMaxentClassifier(MaxEnt maxent){
    this.maxent=maxent;
  }
  
  /** {@inheritDoc} */
  @Override
  public int classify(SparseFeatureVector s) {
    Instance converted = MalletMaxentTrainer.convert(maxent.getAlphabet(), s, "");
    return maxent.classify(converted).getLabeling().getBestIndex();
  }

  /** {@inheritDoc} */
  @Override
  public List<Integer> classifyNBest(int n, SparseFeatureVector s) {
    Instance converted = MalletMaxentTrainer.convert(maxent.getAlphabet(), s, "");
    List<Integer> indices = Lists.newArrayList();
    for (int i=0; i<n; i++){
      indices.add(maxent.classify(converted).getLabeling().getLabelAtRank(i).getIndex());
    }
    return indices;
  }

}
