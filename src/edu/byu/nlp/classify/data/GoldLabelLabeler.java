package edu.byu.nlp.classify.data;

import java.util.List;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;

public class GoldLabelLabeler implements DatasetLabeler{

	@Override
	public Predictions label(Dataset trainingInstances, Dataset heldoutInstances) {

		List<Prediction> labeledPredictions = Lists.newArrayList();
		List<Prediction> unlabeledPredictions = Lists.newArrayList();
		List<Prediction> heldoutPredictions = Lists.newArrayList();

		for (DatasetInstance inst: trainingInstances){
			Preconditions.checkArgument(inst.hasLabel(),"gold labels are not available for instance "+inst.getInfo().getSource());
			if (inst.hasAnnotations()){
				labeledPredictions.add(new BasicPrediction(inst.getLabel(), inst));
			}
			else{
				unlabeledPredictions.add(new BasicPrediction(inst.getLabel(), inst));
			}
		}
		for (DatasetInstance inst: heldoutInstances){
			Preconditions.checkArgument(inst.hasLabel(),"gold labels are not available for instance "+inst.getInfo().getSource());
			heldoutPredictions.add(new BasicPrediction(inst.getLabel(), inst));
		}

		int numAnnotators = trainingInstances.getInfo().getNumAnnotators();
		int numClasses = trainingInstances.getInfo().getNumClasses();
		double[] annotatorAccuracies = new double[numAnnotators];
		double[][][] annotatorConfusionMatrices = new double[numAnnotators][numClasses][numClasses];
		double machineAccuracy = -1;
		double[][] machineConfusionMatrix = new double[numClasses][numClasses];
		double logJoint = -1;
		return new Predictions(labeledPredictions, unlabeledPredictions, heldoutPredictions, 
				annotatorAccuracies, annotatorConfusionMatrices, machineAccuracy, machineConfusionMatrix, logJoint);
		
	}

}
