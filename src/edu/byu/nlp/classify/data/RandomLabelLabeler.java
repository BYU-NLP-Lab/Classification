package edu.byu.nlp.classify.data;

import java.util.List;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.collect.Lists;

import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;

public class RandomLabelLabeler implements DatasetLabeler{

	private RandomGenerator rnd;

	public RandomLabelLabeler(RandomGenerator rnd){
		this.rnd=rnd;
	}
	
	@Override
	public Predictions label(Dataset trainingInstances, Dataset heldoutInstances) {

		List<Prediction> labeledPredictions = Lists.newArrayList();
		List<Prediction> unlabeledPredictions = Lists.newArrayList();
		List<Prediction> heldoutPredictions = Lists.newArrayList();

		for (DatasetInstance inst: trainingInstances){
			int randLabel = rnd.nextInt(trainingInstances.getInfo().getNumClasses());
			if (inst.hasAnnotations()){
				labeledPredictions.add(new BasicPrediction(randLabel, inst));
			}
			else{
				unlabeledPredictions.add(new BasicPrediction(randLabel, inst));
			}
		}
		for (DatasetInstance inst: heldoutInstances){
			int randLabel = rnd.nextInt(trainingInstances.getInfo().getNumClasses());
			heldoutPredictions.add(new BasicPrediction(randLabel, inst));
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
