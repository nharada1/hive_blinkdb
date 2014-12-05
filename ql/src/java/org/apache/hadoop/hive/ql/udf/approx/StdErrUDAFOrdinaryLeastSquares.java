/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.hadoop.hive.ql.udf.approx;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

import org.la4j.LinearAlgebra;
import org.la4j.matrix.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.Vector;
import org.la4j.vector.dense.BasicVector;
import org.la4j.inversion.MatrixInverter;
import org.la4j.linear.LeastSquaresSolver;
import org.la4j.decomposition.MatrixDecompositor;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.lang.SerializationUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StandardListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BinaryObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.DoubleObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.LongObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.StringUtils;

/**
 * GenericUDAFOrdinaryLeastSquares.
 *
 */
@Description(name = "err_approx_ols", value = "_FUNC_(x) - Performs an ordinary least squares regression on a set of columns within a tolerence")
public class StdErrUDAFOrdinaryLeastSquares extends AbstractGenericUDAFResolver {

  static final Log LOG = LogFactory.getLog(StdErrUDAFOrdinaryLeastSquares.class.getName());
  // In the shark driver, we manually intercept queries and add the sample size and 
  // dataset size (in that order) to the end of the query. So when we get it, we get
  // err_approx_ols(X1,X2,...,Xn,Y,SAMPLESIZE,DATASETSIZE). For lack of a better word
  // I'm calling those extra arguments INDUCED_ARGS. They are always at the end.
  // For now, we aren't actually passing induced args, but just in case I'll leave the logic
  private static final int INDUCED_ARGS = 0;

  @Override
  public GenericUDAFEvaluator getEvaluator(TypeInfo[] parameters)
      throws SemanticException {
    if (parameters.length < 2+INDUCED_ARGS) {
     throw new UDFArgumentTypeException(parameters.length-1,
          "Expected at least two arguments but got " + Integer.toString(parameters.length-2));
    }

    // The first argument should be an array, and the 2nd through N-2 should be primitive
    if (parameters[0].getCategory() != ObjectInspector.Category.LIST) {
      throw new UDFArgumentTypeException(0,
        "First argument must be a list type");
    }
    for (int i=1; i<parameters.length-INDUCED_ARGS; i++) {
      if (parameters[i].getCategory() != ObjectInspector.Category.PRIMITIVE) {
        throw new UDFArgumentTypeException(i,
            "Only primitive type arguments are accepted but "
                + parameters[i].getTypeName() + " is passed.");
      }

      switch (((PrimitiveTypeInfo) parameters[i]).getPrimitiveCategory()) {
        case BYTE:
        case SHORT:
        case INT:
        case LONG:
        case FLOAT:
        case DOUBLE:
            break;
        case DATE:
        case TIMESTAMP:
        case STRING:
        case BOOLEAN:
        default:
          throw new UDFArgumentTypeException(0,
              "Only numeric type arguments are accepted but "
                  + parameters[i].getTypeName() + " is passed.");
        }
    }
    return new StdErrUDAFOrdinaryLeastSquaresNumeric();
  }

  /**
   * StdErrUDAFOrdinaryLeastSquaresNumeric.
   *
   */
  public static class StdErrUDAFOrdinaryLeastSquaresNumeric extends GenericUDAFEvaluator {
    // For PARTIAL1 and COMPLETE
    private ArrayList<PrimitiveObjectInspector> inputOI;
    private StandardListObjectInspector bhatOI;

    // For PARTIAL2 and FINAL
    private StructObjectInspector soi;
    private StructField countField;
    private StructField residualField;
    private StructField AField;
    private StructField betaHatField;

    private LongObjectInspector countFieldOI;
    private DoubleObjectInspector residualFieldOI;
    private BinaryObjectInspector AFieldOI;
    private BinaryObjectInspector betaHatFieldOI;

    // For PARTIAL1 and PARTIAL2
    private Object[] partialResult;

    // For FINAL and COMPLETE
    //private ArrayList<Double> result;
    // For FINAL and COMPLETE
    Text result;

    @Override
    public ObjectInspector init(Mode m, ObjectInspector[] parameters) throws HiveException {
      assert (parameters.length >= 2+INDUCED_ARGS);
      super.init(m, parameters);
      
      // init input
      if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {
        inputOI = new ArrayList<PrimitiveObjectInspector>();
        // Add our betaHat
        bhatOI = ((StandardListObjectInspector) parameters[0]);
        // Add our values
        for (int i=1; i<parameters.length-INDUCED_ARGS; i++) {
          inputOI.add((PrimitiveObjectInspector) parameters[i]);
        }
      } else {
        soi = (StructObjectInspector) parameters[0];

        countField = soi.getStructFieldRef("count");
        residualField = soi.getStructFieldRef("residual");
        AField = soi.getStructFieldRef("A");
        betaHatField = soi.getStructFieldRef("betaHat");

        countFieldOI = (LongObjectInspector) countField
            .getFieldObjectInspector();
        residualFieldOI = (DoubleObjectInspector) residualField
            .getFieldObjectInspector();
        AFieldOI = (BinaryObjectInspector) AField
            .getFieldObjectInspector();
        betaHatFieldOI = (BinaryObjectInspector) betaHatField 
            .getFieldObjectInspector();
      }

      // init output
      if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {
        // The output of a partial aggregation is a struct containing
        // a long count and doubles sum and variance.

        ArrayList<ObjectInspector> foi = new ArrayList<ObjectInspector>();

        foi.add(PrimitiveObjectInspectorFactory.writableBooleanObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableBinaryObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableBinaryObjectInspector);

        ArrayList<String> fname = new ArrayList<String>();
        fname.add("empty");
        fname.add("count");
        fname.add("residual");
        fname.add("A");
        fname.add("betaHat");

        partialResult = new Object[5];
        partialResult[0] = new BooleanWritable();
        partialResult[1] = new LongWritable(0);
        partialResult[2] = new DoubleWritable(0.0);
        partialResult[3] = new BytesWritable();
        partialResult[4] = new BytesWritable(); 

        return ObjectInspectorFactory.getStandardStructObjectInspector(fname,
            foi);

      } else {

        result = new Text();
        return PrimitiveObjectInspectorFactory.writableStringObjectInspector;

    //    result = new ArrayList(); 
//        return ObjectInspectorFactory.getStandardListObjectInspector(
  //                  PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
      }

    }

    /** class for storing double sum value. */
    static class SumDoubleAgg implements AggregationBuffer {
      boolean empty;
      long count; // number of elements
      double residual;
      Matrix A;
      ArrayList<Double> betaHat;
    }

    @Override
    public AggregationBuffer getNewAggregationBuffer() throws HiveException {
      SumDoubleAgg result = new SumDoubleAgg();
      reset(result);
      return result;
    }

    @Override
    public void reset(AggregationBuffer agg) throws HiveException {
      SumDoubleAgg myagg = (SumDoubleAgg) agg;
      myagg.empty = true;
      myagg.count = 0;
      myagg.A = null;
      myagg.residual = 0.0;
      myagg.betaHat = new ArrayList<Double>(); 
    }

    boolean warned = false;

    @Override
    public void iterate(AggregationBuffer agg, Object[] parameters) throws HiveException {
      assert (parameters.length >= 2+INDUCED_ARGS);
      
      if (parameters.length >= 2+INDUCED_ARGS) {
          // Convert the ArrayList to real doubles and not DoubleWritables
          ArrayList betaHatDW = new ArrayList(bhatOI.getList(parameters[0]));
          ArrayList<Double> betaHat = new ArrayList<Double>();
          for (int i=0; i<betaHatDW.size(); i++) {
            Object bvDW = betaHatDW.get(i);
            PrimitiveObjectInspector thisOI = (PrimitiveObjectInspector) bhatOI.getListElementObjectInspector();
            double bVald = PrimitiveObjectInspectorUtils.getDouble(bvDW, thisOI);
            betaHat.add(bVald);
          } 
          ((SumDoubleAgg) agg).betaHat = betaHat;
      }

      boolean nulls = false;
      for (int i=0; i<parameters.length; i++) {
        if (parameters[i] == null) {
          nulls = true;
        }
      }
      if (!nulls) {
        Object p = new Object();
        
        int nParams = parameters.length-INDUCED_ARGS;
        SumDoubleAgg myagg = (SumDoubleAgg) agg;
        try {
          // Parameters should be [Array betaHat, x1, x2, ..., xn, y]

          // Our vA vector should have [x1,x2,...,xn,y]
          // Note that we have to offset our indicies because vA and thisOI are missing
          // the first bHat parameter value
          double[] vA = new double[nParams-1];
          for (int i=1; i<nParams; i++) { 
            p = parameters[i]; 
            PrimitiveObjectInspector thisOI = inputOI.get(i-1);
            vA[i-1] = PrimitiveObjectInspectorUtils.getDouble(p, thisOI);
          }

          // Generate the A matrix
          // Two less because we ignore the last column of input data as well as the first (coeffs)
          double[][] clipped_mat = new double[1][nParams-2]; 
          for (int i=0; i<nParams-2; i++)
            clipped_mat[0][i] = vA[i];
          // A = X'*X, where in this case X is a vector containing one row of information
          Matrix this_X = new Basic2DMatrix(clipped_mat);
          Matrix this_XT = this_X.transpose();
          Matrix this_A = this_XT.multiply(this_X);
          // We have to test for empty A. This is because during our constructor we 
          // can't access the number of parameters in the query
          if (myagg.A == null) {
            myagg.A = this_A;
          } else {
            myagg.A = myagg.A.add(this_A);
          }
          
          // Get our value of Y for this row also, which is the last value of vA
          double this_y = vA[vA.length-1]; 
        
          // Calculate this residual
          double residual = this_y;
          for (int i=0; i<myagg.betaHat.size(); i++) {
            Double bhVal = (Double) myagg.betaHat.get(i);
            LOG.warn(vA.toString());
            residual -= vA[i]*bhVal;
          }  
    
          // Add residual to the aggregate
          myagg.residual += residual*residual;
          // Increase the count         
          myagg.count++;
        } catch (NumberFormatException e) {
          if (!warned) {
            warned = true;
            LOG.warn(getClass().getSimpleName() + " "
                + StringUtils.stringifyException(e));
            LOG
            .warn(getClass().getSimpleName()
                + " ignoring similar exceptions.");
          }
        }
      }
    }

    @Override
    public Object terminatePartial(AggregationBuffer agg) throws HiveException {
      // return terminate(agg);
      SumDoubleAgg myagg = (SumDoubleAgg) agg;
      ((BooleanWritable) partialResult[0]).set(myagg.empty);
      ((LongWritable) partialResult[1]).set(myagg.count);
      ((DoubleWritable) partialResult[2]).set(myagg.residual);
    
      // We have to serialize our matricies to bytes. We do this so that we can live on disk between
      // map and reduce phases under the hood (especially if we can't live in memory).      
      byte[] serializedA = SerializationUtils.serialize(myagg.A);
      ((BytesWritable) partialResult[3]).set(serializedA, 0, serializedA.length);

      byte[] serializedbetaHat = SerializationUtils.serialize(myagg.betaHat);
      ((BytesWritable) partialResult[4]).set(serializedbetaHat, 0, serializedbetaHat.length);
      return partialResult;
    }

    @Override
    public void merge(AggregationBuffer agg, Object partial) throws HiveException {

      if (partial != null) {

        SumDoubleAgg myagg = (SumDoubleAgg) agg;
        myagg.empty = false;

        Object partialCount = soi.getStructFieldData(partial, countField);
        Object partialResidual = soi.getStructFieldData(partial, residualField);
        Object partialA = soi.getStructFieldData(partial, AField);
        Object partialBetaHat = soi.getStructFieldData(partial, betaHatField);

        long n = myagg.count;
        long m = countFieldOI.get(partialCount);
        double r = residualFieldOI.get(partialResidual);
        // To deserialize we have to call getData because the get method returns a ByteArrayRef
        // in an attempt to improve performance, however at the time of writing this is undocumented.
        byte[] des_A = AFieldOI.getPrimitiveJavaObject(partialA).getData();
        Matrix A = (Basic2DMatrix) SerializationUtils.deserialize(des_A); 
    
        byte[] des_betaHat = betaHatFieldOI.getPrimitiveJavaObject(partialBetaHat).getData();
        myagg.betaHat = (ArrayList) SerializationUtils.deserialize(des_betaHat); 

        if (n == 0) {
          // Just copy the information since there is nothing so far
          myagg.count = m; 
          myagg.A = A;
          myagg.residual = r;
        }

        if (m != 0 && n != 0 && r != 0 && A != null) {
          // Merge the two partials
          myagg.empty = false;
          myagg.count += m;
          myagg.residual += r;
          myagg.A = myagg.A.add(A);
        }
      }
    }

    @Override
    public Object terminate(AggregationBuffer agg) throws HiveException {
      SumDoubleAgg myagg = (SumDoubleAgg) agg;
      if (myagg.empty) {
        return null;
      }
      
      StringBuilder sb = new StringBuilder();

      // inv(X'X) or inv(A)
      MatrixInverter inverter = myagg.A.withInverter(LinearAlgebra.GAUSS_JORDAN);
      Matrix Ainv = inverter.inverse(LinearAlgebra.DENSE_FACTORY);
      // rows, cols, and normalization
      long rows = myagg.count;
      int cols = myagg.A.columns(); 
      double degrees_of_freedom = (double)(rows-cols);
      double norm_factor = 1/degrees_of_freedom;
      // confidence interval
      // THIS NEEDS TO BE STUDENT T FOR LOW DEGREES OF FREEDOM FIX THIS YOU FUCKING IDIOT
      TDistribution distr = new TDistribution(degrees_of_freedom);
      double qN = distr.inverseCumulativeProbability(0.995); //Two sided 99%
      // calculate the interval for each beta
      double[] interval = new double[myagg.betaHat.size()];
      for (int j=0; j<myagg.betaHat.size(); j++) {
        interval[j] = qN * Math.sqrt(norm_factor * Ainv.get(j,j) * myagg.residual);
      }

      /*double bhat[] = solved.toArray();
      // Convert the array list to the DoubleWritable type
      ArrayList<DoubleWritable> result = new ArrayList<DoubleWritable>();
      for (double d : bhat)
        result.add(new DoubleWritable(d)); */

      sb.append("Count: ");
      sb.append(myagg.count);
      sb.append("\nA: ");
      sb.append(myagg.A.toString());
      sb.append("residual: ");
      sb.append(myagg.residual);
      sb.append("\nNorm Factor: ");
      sb.append(norm_factor);

      sb.append("\n\nError Bounds:\n\n");
      for (int j=0; j<myagg.betaHat.size(); j++) {
        sb.append(myagg.betaHat.get(j));
        sb.append(" +/- ");
        sb.append(interval[j]);
        sb.append("\n");
      }
      sb.append("\n\nwith 99% Confidence");
      result.set(sb.toString());
      return result;

    }
  }
}
