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

import java.util.Arrays;

import com.google.common.base.Preconditions;

import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.AbstractRealMatrixPreservingVisitor;
import edu.byu.nlp.util.ColumnMajorMatrices;
import edu.byu.nlp.util.DoubleArrays;


/**
 * Add one smoothing.
 * @author rah67
 *
 */
public class UncertaintyPreservingNaiveBayesLearner implements ClassifierLearner {

	/** {@inheritDoc} */
	@Override
	public NaiveBayesClassifier learnFrom(final Dataset data) {
		Preconditions.checkNotNull(data);
		Preconditions.checkArgument(data.getInfo().getNumClasses() > 0, "Dataset must have at least one class");
		Preconditions.checkArgument(data.getInfo().getNumFeatures() >= 0, "Dataset must have zero or more features");
		
		final double[] weights = new double[data.getInfo().getNumClasses() * data.getInfo().getNumFeatures()];
		final double[] biases = new double[data.getInfo().getNumClasses()];

		Dataset labeledData = Datasets.divideInstancesWithObservedLabels(data).getFirst();
		
		// Add-one smoothing.
		Arrays.fill(weights, 1.0);
		Arrays.fill(biases, 1.0);
		
		// data counts (count ALL annotations)
		for (final DatasetInstance instance : labeledData) {
		  String src = instance.getInfo().getRawSource();
		  // instance has a label -- use that
		  if (instance.getObservedLabel()!=null){
	        int label = instance.getObservedLabel(); 
	        //if (label==0){System.out.println("label 0: "+src);}
			biases[label] += 1;
	        instance.asFeatureVector().scaleAndAddToRow(weights, label, data.getInfo().getNumClasses(), 1);
		  }
		  // FIXME: currently this class is only called by SingleLabelLabeler, which calls it on data that has 
		  // already been through DataBuilder converting all annotations to single labels. This must be fixed 
		  // before this class is able to take advantage of annotation information 
		  
		  // instance only has annotations -- use them (down-weighted so that together they sum to 1)
		  else{
		    double numAnnotations = instance.getInfo().getNumAnnotations();
		    final double value = 1.0 / numAnnotations;
  		  Preconditions.checkArgument(numAnnotations>0);
  		  
  		  instance.getAnnotations().getLabelAnnotations().walkInOptimizedOrder(new AbstractRealMatrixPreservingVisitor() {
          @Override
          public void visit(int annotator, int annval, double numanns) {
            instance.asFeatureVector().scaleAndAddToRow(weights, annval, data.getInfo().getNumClasses(), value*numanns);
          }
        });
  		  
		  }
		}

		// Compute log p(c)
		DoubleArrays.logToSelf(biases);
		DoubleArrays.logNormalizeToSelf(biases);
		
		// Compute log p(f|c)
		ColumnMajorMatrices.normalizeRows(weights, biases.length);
		DoubleArrays.logToSelf(weights);
		
		return new NaiveBayesClassifier(biases, weights);
	}

}
