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
package edu.byu.nlp.classify.data;

import java.util.List;

import org.apache.commons.math3.linear.SparseRealMatrix;

import edu.byu.nlp.math.SparseRealMatrices;

/**
 * @author pfelt
 * 
 * Takes care of the logic associated with detecting 
 * instances with no annotations and returning the null label
 */
public abstract class AbstractLabelChooser implements LabelChooser{

  protected abstract List<Integer> labelsForNonEmpty(SparseRealMatrix annotations);
  
  /** {@inheritDoc} */
  @Override
  public List<Integer> labelsFor(SparseRealMatrix annotations) {
    if (annotations==null || SparseRealMatrices.sum(annotations)==0){
      return null;
    }
    return labelsForNonEmpty(annotations);
  }

}
