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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;


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
  public static final int MAXIMIZE_MAX_ITERATIONS = 20;
  public static final int MAXIMIZE_BATCH_SIZE = 3;

  
  public interface SupportsTrainingOperations{
    void sample(String variableName, String[] args);
    void maximize(String variableName, String[] args);
  }
  
  public interface Operation{
    void doOperation(SupportsTrainingOperations model);
  }
  
  public static enum OperationType{
    NONE,
    SAMPLE,
    MAXIMIZE
  }
  
  public static class OperationParser{
    public static final String OUTER_DELIM = ":";
    public static final String INNER_DELIM = "-";
    
    public static Iterable<Operation> parse(String ops){
      List<Operation> parsedOps = Lists.newArrayList();
      for (String op: ops.split(OUTER_DELIM)){
        logger.info("Doing training operation "+op);
        parsedOps.add(parseInner(op));
      }
      return parsedOps;
    }
    
    public static Operation parseInner(String rawOp){
      String[] fields = rawOp.split(INNER_DELIM);
      Preconditions.checkState(fields.length>=1,"training operation must contain at least an operation name from {sample,maximize}"); // require at LEAST one field
      OperationType type = OperationType.valueOf(fields[0].toUpperCase());
      final String variableName = fields.length>=2? fields[1]: null;
      final String[] args = fields.length>=3? Arrays.copyOfRange(fields, 2, fields.length): new String[]{};
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
            model.maximize(variableName, args); 
          }
        };
      case SAMPLE:
        return new Operation(){
          @Override
          public void doOperation(SupportsTrainingOperations model) {
            model.sample(variableName, args); 
          }
        };
      default:
        throw new UnsupportedOperationException("Unknown operation type "+type);
      }
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
    logger.info("Training operations "+ops);
    for (Operation op: OperationParser.parse(ops)){
      op.doOperation(model);
    }
  }

  
}
