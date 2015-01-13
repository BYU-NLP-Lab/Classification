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
package edu.byu.nlp.classify.eval;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import edu.byu.nlp.util.Indexer;

public class ConfusionMatrix {

  private final RealMatrix matrix;
  private final Indexer<String> labels;

  public ConfusionMatrix(int numLabels, int K, Indexer<String> labels) {
      this.labels = labels;
      this.matrix = new Array2DRowRealMatrix(numLabels, K);
  }

  public void incrementEntry(int truth, int predicted) {
      addToEntry(truth, predicted, 1);
  }

  public void addToEntry(int truth, int predicted, int amount) {
      matrix.addToEntry(truth, predicted, amount);
  }

  public double[][] getData(){
    return matrix.getData();
  }

  @Override
  public String toString() {
      int maxLen = 0;
      for (String label : labels) {
          if (maxLen < label.length()) {
              maxLen = label.length();
          }
      }

      String title = "Truth\\Pred";
      maxLen = Math.max(maxLen, title.length());

      String headerFormat = "%" + maxLen + "s";
      StringBuilder sb = new StringBuilder(String.format(headerFormat, title));

      for (int i = 0; i < matrix.getColumnDimension(); i++) {
          sb.append(String.format(" %6d", i));
      }
      sb.append('\n');

      int rowLength = maxLen + 7 * labels.size();
      for (int i = 0; i < rowLength; i++) {
          sb.append('-');
      }
      sb.append('\n');

      for (int truth = 0; truth < matrix.getRowDimension(); truth++) {
          sb.append(String.format(headerFormat, labels.get(truth)));
          for (int guess = 0; guess < matrix.getColumnDimension(); guess++) {
              sb.append(String.format(" %6d", (int) matrix.getEntry(truth, guess)));
          }
          sb.append('\n');
      }
      return sb.toString();
  }

}

