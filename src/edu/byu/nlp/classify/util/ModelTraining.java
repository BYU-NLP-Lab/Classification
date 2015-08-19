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
package edu.byu.nlp.classify.util;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import edu.byu.nlp.classify.data.DatasetLabeler;


/**
 * @author pfelt
 * 
 * Utilities to parse a string (probably passed in via a command line argument) 
 * that specifies how a model ought to be trained. For example, input string takes 
 * the form sample-1-3:sample 
 *
 */
public class ModelTraining {

  private static final Logger logger = LoggerFactory.getLogger(ModelTraining.class);
  public static final double MAXIMIZE_IMPROVEMENT_THRESHOLD = 1e-6;
  public static final int MAXIMIZE_MAX_ITERATIONS = 100;
  public static final int MAXIMIZE_BATCH_SIZE = 3;

  
  public interface SupportsTrainingOperations{
    Double sample(String variableName, int iteration, String[] args);
    Double maximize(String variableName, int iteration, String[] args);
    // optional method for debugging (e.g., visualize training by printing a progression of confusion matrices as training progresses)
    DatasetLabeler getIntermediateLabeler();
  }
  
  public interface Operation{
    void doOperation(SupportsTrainingOperations model);
  }
  
  public interface IntermediatePredictionLogger{
    void logPredictions(int iteration, OperationType opType, String variableName, String[] args, DatasetLabeler intermediateLabeler);
  }
  
  public static enum OperationType{
    NONE,
    SAMPLE,
    MAXIMIZE
  }
  
  public static class OperationParser{
    public static final String OUTER_DELIM = ":";
    public static final String INNER_DELIM = "-";
    private OperationExecutor executor;
    
    public OperationParser(){
      this(null);
    }
    public OperationParser(IntermediatePredictionLogger predictionLogger){
      this.executor = new OperationExecutor(predictionLogger);
    }
    
    public Iterable<Operation> parse(String ops){
      List<Operation> parsedOps = Lists.newArrayList();
      for (String op: ops.split(OUTER_DELIM)){
        logger.info("Doing training operation "+op);
        parsedOps.add(parseInner(op));
      }
      return parsedOps;
    }
    
    public Operation parseInner(String rawOp){
      String[] fields = rawOp.split(INNER_DELIM);
      Preconditions.checkState(fields.length>=1,"training operation must contain at an operation (e.g., maximize, sample, none)"); 
      OperationType type = OperationType.valueOf(fields[0].toUpperCase());
      final String variableName = fields.length>=2? fields[1]: "all"; // default to "all"
      final Integer iterations = fields.length>=3? parseInt(fields[2],"Number of Iterations must be an integer!"): null;
      final String[] args = fields.length>=3? Arrays.copyOfRange(fields, 3, fields.length): new String[]{};
      switch(type){
      case NONE:
        return new Operation() {
          @Override
          public void doOperation(SupportsTrainingOperations model) {
            // do nothing
          }
        };
      case MAXIMIZE:
        return new Operation(){
          @Override
          public void doOperation(SupportsTrainingOperations model) {
            executor.maximize(model, variableName, iterations, args);
          }
        };
      case SAMPLE:
        return new Operation(){
          @Override
          public void doOperation(SupportsTrainingOperations model) {
            executor.sample(model, variableName, iterations, args);
          }
        };
      default:
        throw new UnsupportedOperationException("Unknown operation type "+type);
      }
    }
  }


  private static Integer parseInt(String str, String errorMessage){
    try{
      return Integer.parseInt(str);
    }
    catch(Exception e){
      throw new IllegalArgumentException("Unable to parse integer value from "+str+". "+errorMessage);
    }
  }
  
  
  public static class OperationExecutor{

    private double maximizationImprovementThreshold = MAXIMIZE_IMPROVEMENT_THRESHOLD;
    private int maxNumIterations = MAXIMIZE_MAX_ITERATIONS;
    private IntermediatePredictionLogger predictionLogger;

    public OperationExecutor(){
      this(null);
    }
    public OperationExecutor(IntermediatePredictionLogger predictionLogger){
      this.predictionLogger = predictionLogger;
    }
    public void setMaximizationImprovementThreshold(double threshold){
      maximizationImprovementThreshold=threshold;
    }
    public void setMaxNumIterations(int iterations){
      maxNumIterations=iterations;
    }
    
    private void maximize(SupportsTrainingOperations model, String variableName, Integer iterations, String[] args){
      Double value = null;
      // maximize until convergence if no iterations are specified
      if (iterations==null || iterations==0){
        value = maximizeUntilConvergence(model, maximizationImprovementThreshold, maxNumIterations,  variableName, args);
      }
      // maximize for the specified number of iterations 
      else{
        for (int i=0; i<iterations; i++){
          value = model.maximize(variableName, i, args);
          if (value!=null){
            logger.info("maximize-"+variableName+" (args="+Joiner.on('-').join(args)+" iterations="+i+") with value (probably unnormalized log joint) "+value);
          }
          if (predictionLogger!=null){
            predictionLogger.logPredictions(i, OperationType.MAXIMIZE, variableName, args, model.getIntermediateLabeler());
          }
        }
      }
      logger.info("finished sampling "+variableName+" (args="+Joiner.on('-').join(args)+" iterations="+iterations+") with value (probably unnormalized log joint) "+value);
    }
    
    private void sample(SupportsTrainingOperations model, String variableName, Integer iterations, String[] args){
      Double value = null;

      // sample until convergence if no iterations are specified
      if (iterations==null || iterations==0){
        throw new IllegalArgumentException("No automatic convergence criterion implemented for sampling. You must specify a number of iterations.");
      }
      // sample for the specified number of iterations 
      for (int i=0; i<iterations; i++){
        value = model.sample(variableName, i, args);
        if (value!=null){
          logger.info("sample-"+variableName+" (args="+Joiner.on('-').join(args)+" iterations="+i+") with value (probably unnormalized log joint) "+value);
        }
        if (predictionLogger!=null){
          predictionLogger.logPredictions(i, OperationType.SAMPLE, variableName, args, model.getIntermediateLabeler());
        }
      }
      logger.info("finished sample-"+variableName+" (args="+Joiner.on('-').join(args)+" iterations="+iterations+") with value (probably unnormalized log joint) "+value);
    }
    
    private Double maximizeUntilConvergence(SupportsTrainingOperations model, double minChange, int maxNumIterations, String variableName, String[] args) {
      double change = Double.MAX_VALUE;
      double prevVal = -Double.MAX_VALUE;
      int i = 0;
//      while (change > minChange && i < maxNumIterations){
      while (i < maxNumIterations){
        Double currVal = model.maximize(variableName, i, args);
        if (currVal!=null){
          change = currVal - prevVal;
          prevVal = currVal;
          logger.debug("maximize-"+variableName+" (args="+Joiner.on('-').join(args)+" iteration="+i+") with a value of "+currVal+" (improvement of "+change+")");
          if (predictionLogger!=null){
            predictionLogger.logPredictions(i, OperationType.MAXIMIZE, variableName, args, model.getIntermediateLabeler());
          }
        }
        else{
          logger.debug("maximize-"+variableName+" (args="+Joiner.on('-').join(args)+" iteration="+i);
          change = Double.MAX_VALUE; // we were given no value by which to judge convergence this time
        }
        
        i++;
      }
      logger.info("finished maximization after "+i+" iterations");
      return prevVal;
    }
  }
  
  /**
   * Parse and then perform a sequence of colon-delimited training operations with valid values 
   * {sample,maximize,none} where 
   * operations take hyphen-delimited arguments. 
   * For example, sample-m-1-1:maximize:maximize-y will 
   * 1) call sample with variable name="m" and args ["1","1"] 
   * 2) call maximize with variable name=null and args []
   * 3) call maximize with variable name="y" and args []
   */
  public static void doOperations(String ops, SupportsTrainingOperations model){
    doOperations(ops, model, null);
  }
  public static void doOperations(String ops, SupportsTrainingOperations model, IntermediatePredictionLogger predictionLogger){
    logger.info("Training operations "+ops);
    for (Operation op: new OperationParser(predictionLogger).parse(ops)){
      op.doOperation(model);
    }
  }

  
}
