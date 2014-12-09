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
import java.util.Arrays;
import java.util.*; //added
import java.lang.*; //add for squareroot

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
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector; //added
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
@Description(name = "test_approx_ols", 
          value = "_FUNC_(x) - Predicts values of a column based on a set of regression inputs",
          extended = "Example:\n"
          + "SELECT test_approx_ols(beta array, A^(-1) array, s^2 scalar, x0 (key column), x1,x2,x3) FROM test;\n"
          + "beta is a column of size 3 of regression estimates computed from approx_ols_test."
          + "A^(-1) is an array of size 3^2 given by prep_test_approx_ols"
          + "s^2 is an estimate of s^2 given by prep_test_approx_ols"
          + "x0 is a unique identifier of each row (e.g., UMID in a table of umich students"
          + "x1,x2,x3 is a set of columns from the table test with the same number of rows."
          + "It returns a map that associates the key of each row with the predicted value of the row.")

//TODO: need to make it check that the dimension of beta is the same as the number of predictors (i.e., x1, ...,xn)
public class TestApproxUDAFOrdinaryLeastSquares extends AbstractGenericUDAFResolver {

  static final Log LOG = LogFactory.getLog(TestApproxUDAFOrdinaryLeastSquares.class.getName());

  @Override
  public GenericUDAFEvaluator getEvaluator(TypeInfo[] parameters)
      throws SemanticException {
    if (parameters.length < 5) { //is this the right number of coefficients?
      throw new UDFArgumentTypeException(parameters.length - 1,
          "Expected at least five arguments but got " + parameters.length);
    }

    for (int i=3; i<parameters.length; i++) { //ask nate if it makes sense to start at 4. Need to skip over first few args
      if (parameters[i].getCategory() != ObjectInspector.Category.PRIMITIVE) {
        throw new UDFArgumentTypeException(0,
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
    return new TestApproxUDAFOrdinaryLeastSquaresNumeric();
  }

  /**
   * ApproxUDAFOrdinaryLeastSquaresTestNumeric.
   *
   */
  public static class TestApproxUDAFOrdinaryLeastSquaresNumeric extends GenericUDAFEvaluator {
    // For PARTIAL1 and COMPLETE: ObjectInspectors for original data
    private ArrayList<PrimitiveObjectInspector> inputOI;
    private StandardListObjectInspector bhatOI;
    private StandardListObjectInspector AinvOI; //I need another listobjectinspector for ainv, correct?
    private DoubleObjectInspector sOI; //is this correct?


    // For PARTIAL2 and FINAL: ObjectInspectors for partial aggregations
    private StructObjectInspector soi;

    private StructField countField;
    private StructField predField;
    private StructField ciField;
    private StructField betaHatField;
    private StructField AinvField;
    private StructField sField;

    private LongObjectInspector countFieldOI;
    private BinaryObjectInspector predFieldOI;
    private BinaryObjectInspector ciFieldOI;
    private BinaryObjectInspector betaHatFieldOI;
    private BinaryObjectInspector AinvFieldOI;
    private DoubleObjectInspector sFieldOI;

    // For PARTIAL1 and PARTIAL2
    private Object[] partialResult;

    // For FINAL and COMPLETE
    // private HashMap<Double, Double> result; //should this be upper case double, i.e., Double
    Text result;

    @Override
    public ObjectInspector init(Mode m, ObjectInspector[] parameters) throws HiveException {
      assert (parameters.length >= 2);
      super.init(m, parameters);

      // init input
      if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {
        inputOI = new ArrayList<PrimitiveObjectInspector>();
        // Add our betaHat
        bhatOI = ((StandardListObjectInspector) parameters[0]); 
        AinvOI = ((StandardListObjectInspector) parameters[1]);
        sOI = ((DoubleObjectInspector) parameters[2]);
        for (int i=3; i<parameters.length; i++) {
          inputOI.add((PrimitiveObjectInspector) parameters[i]);
        }
      } else {
        soi = (StructObjectInspector) parameters[0];

        countField = soi.getStructFieldRef("count");
        predField = soi.getStructFieldRef("pred");
        ciField = soi.getStructFieldRef("ci");
        betaHatField = soi.getStructFieldRef("betaHat");
        AinvField = soi.getStructFieldRef("Ainv");
        sField = soi.getStructFieldRef("s");

        countFieldOI = (LongObjectInspector) countField
            .getFieldObjectInspector();
        predFieldOI = (BinaryObjectInspector) predField
            .getFieldObjectInspector();
        ciFieldOI = (BinaryObjectInspector) ciField
            .getFieldObjectInspector();
        betaHatFieldOI = (BinaryObjectInspector) betaHatField 
            .getFieldObjectInspector();
        AinvFieldOI = (BinaryObjectInspector) AinvField 
            .getFieldObjectInspector();
        sFieldOI = (DoubleObjectInspector) sField
            .getFieldObjectInspector();
      }

      // init output
      if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {
        // The output of a partial aggregation is a struct containing
        // a long count and doubles sum and variance.

        ArrayList<ObjectInspector> foi = new ArrayList<ObjectInspector>();

      //what we need here:
      // boolean empty;
      // long count; // number of elements
      // ArrayList<Double> betaHat;
      // HashMap<Double, Double> pred; //predicted value by column key
      // HashMap<Double, Double> ci //confidence interval value
      // Arraylist<Double> Ainv; //Ainv array that must be reshaped to obtain square form
      // Double s; //estimate of sigmasquared

        foi.add(PrimitiveObjectInspectorFactory.writableBooleanObjectInspector); //what is foi for?
        foi.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableBinaryObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableBinaryObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableBinaryObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableBinaryObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);



        ArrayList<String> fname = new ArrayList<String>();
        fname.add("empty");
        fname.add("count");
        fname.add("betaHat");
        fname.add("pred");
        fname.add("ci");
        fname.add("Ainv");
        fname.add("s");

        partialResult = new Object[7];
        partialResult[0] = new BooleanWritable();
        partialResult[1] = new LongWritable(0);
        partialResult[2] = new BytesWritable();
        partialResult[3] = new BytesWritable();
        partialResult[4] = new BytesWritable();
        partialResult[5] = new BytesWritable();
        partialResult[6] = new DoubleWritable(0);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fname,
            foi);

      } else {
        //suffices to return string outcome like in Nate's function
        result = new Text();
        return PrimitiveObjectInspectorFactory.writableStringObjectInspector;


      }

    }

    /** class for storing betaHat */
    static class SumDoubleAgg implements AggregationBuffer {
      boolean empty;
      long count; // number of elements
      ArrayList<Double> betaHat;
      HashMap<Double, Double> pred; //predicted value by column key
      HashMap<Double, Double> ci; //confidence interval by column key
      ArrayList<Double> Ainv; //Ainv array that must be reshaped to obtain square form
      Double s; //estimate of sigmasquared
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
      myagg.betaHat = new ArrayList<Double>(); 
      myagg.pred = new HashMap<Double, Double>();
      myagg.ci = new HashMap<Double, Double>(); 
      myagg.Ainv = new ArrayList<Double>();
      myagg.s = 0.0; //is this an okay way to instantiate s??
    }

    boolean warned = false;

    @Override
    public void iterate(AggregationBuffer agg, Object[] parameters) throws HiveException {
      assert (parameters.length >= 5); 

      if (parameters.length >= 5) {
          // Convert the ArrayList for betaHat to real doubles and not DoubleWritables
          ArrayList betaHatDW = new ArrayList(bhatOI.getList(parameters[0]));
          ArrayList<Double> betaHat = new ArrayList<Double>();
          for (int i=0; i<betaHatDW.size(); i++) {
            Object bvDW = betaHatDW.get(i);
            PrimitiveObjectInspector thisOI = (PrimitiveObjectInspector) bhatOI.getListElementObjectInspector();
            double bVald = PrimitiveObjectInspectorUtils.getDouble(bvDW, thisOI);
            betaHat.add(bVald);
          } 
          ((SumDoubleAgg) agg).betaHat = betaHat;

          // Convert the ArrayList for Ainv to real doubles and not DoubleWritables
          ArrayList AinvDW = new ArrayList(AinvOI.getList(parameters[1]));
          ArrayList<Double> Ainv = new ArrayList<Double>();
          for (int i=0; i<AinvDW.size();i++) {
            Object avDW = AinvDW.get(i);
            PrimitiveObjectInspector thisOI = (PrimitiveObjectInspector) AinvOI.getListElementObjectInspector();
            double AinvVald = PrimitiveObjectInspectorUtils.getDouble(avDW, thisOI);
            Ainv.add(AinvVald);
          }
          ((SumDoubleAgg) agg).Ainv = Ainv;

          //Obtain Shat Nate: please add code to include Shat here.
          ((SumDoubleAgg) agg).s = PrimitiveObjectInspectorUtils.getDouble(parameters[2],
              sOI);
      }

      boolean nulls = false;
      for (int i=0; i<parameters.length; i++) {
          if (parameters[i] == null) {
              nulls = true;
          }
      }
      if (!nulls) {
        Object p = new Object();
        int nParams = parameters.length;
        SumDoubleAgg myagg = (SumDoubleAgg) agg;
        try {

          // Parameters should be [Array betaHat, array A^(-1), s^2, key column, x1, x2, ..., xn]

          // Our vA vector should have [key column,x1,x2,...,xn]
          // Note that we have to offset our indicies because vA and thisOI are missing
          // the first bHat parameter value

          double[] vA = new double[nParams-3];
          for (int i=3; i<nParams; i++) {
            p = parameters[i];
            PrimitiveObjectInspector thisOI = inputOI.get(i-3); //did I mess this up?
            vA[i-3] = PrimitiveObjectInspectorUtils.getDouble(p, thisOI);
          }

          //check that the number of covariates is equal to the dimension of betahat
          // if (myagg.betaHat.size() != (vA.length-1)) {
          //     throw new HiveException("Coefficient vector has size " + myagg.betaHat.size() + "but should have size" + vA.length);
          // }
  

          // Calculate the dot product of betahat and x1,...,xn; this gives pred
          double pred = 0;
          for (int i=0; i<myagg.betaHat.size(); i++) {
            Double bhVal = (Double) myagg.betaHat.get(i);
            pred += vA[i+1]*bhVal;
          }  
          
          // put pred and the key in the myagg.pred map
          // note that vA[0] should contain the key value
          myagg.pred.put(vA[0], pred);

          //
          //calculate the confidence interval
          //

          //put A^-1 in matrix form
          int m = myagg.Ainv.size();
          // LOG.warn(m);
          double[][] clipped_mat = new double[1][m]; 
          for (int i=0; i<m; i++) {
            clipped_mat[0][i] = myagg.Ainv.get(i); //I don't get this error??
          }
          Matrix this_Ainv = new Basic2DMatrix(clipped_mat);
          int n = (int) Math.sqrt(m);
          // LOG.warn(n);
          Matrix res_Ainv = (Basic2DMatrix) this_Ainv.resize(n, n);

          //make vector of [x1,x2,...,xn]
          int k = vA.length;
          // LOG.warn(k);
          double[][] for_X = new double[k-1][1];
          for (int i=1; i<k; i++) {
            for_X[i-1][0] = vA[i]; //I don't get this error??
          }
          Matrix this_X = new Basic2DMatrix(for_X);
          Matrix this_XT = this_X.transpose();
          Matrix cipart = this_XT.multiply(res_Ainv);
          Matrix ci_mat = cipart.multiply(this_X);
          Double ci = (Double) ci_mat.get(0,0);

          LOG.warn(ci);
          //insert ci into myagg
          myagg.ci.put(vA[0], ci);

          //increase the count
          myagg.count++;
          // LOG.warn(myagg.count);
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

      // We have to serialize our matricies to bytes. We do this so that we can live on disk between
      // map and reduce phases under the hood (especially if we can't live in memory).      
      byte[] serializedbetaHat = SerializationUtils.serialize(myagg.betaHat);
      byte[] serializedpred = SerializationUtils.serialize(myagg.pred);
      byte[] serializedci = SerializationUtils.serialize(myagg.ci);
      byte[] serializedAinv = SerializationUtils.serialize(myagg.Ainv);

      ((BytesWritable) partialResult[2]).set(serializedbetaHat, 0, serializedbetaHat.length);
      ((BytesWritable) partialResult[3]).set(serializedpred, 0, serializedpred.length);
      ((BytesWritable) partialResult[4]).set(serializedci, 0, serializedci.length);
      ((BytesWritable) partialResult[5]).set(serializedAinv, 0, serializedAinv.length);

      ((DoubleWritable) partialResult[6]).set(myagg.s);

      return partialResult;
    }

    @Override
    public void merge(AggregationBuffer agg, Object partial) throws HiveException {

      if (partial != null) {

        SumDoubleAgg myagg = (SumDoubleAgg) agg;
        myagg.empty = false;

        Object partialCount = soi.getStructFieldData(partial, countField);
        Object partialBetaHat = soi.getStructFieldData(partial, betaHatField);
        Object partialpred = soi.getStructFieldData(partial, predField);
        Object partialci = soi.getStructFieldData(partial, ciField);        
        Object partialAinv = soi.getStructFieldData(partial, AinvField);
        Object partials = soi.getStructFieldData(partial, sField);

        long n = myagg.count;
        long m = countFieldOI.get(partialCount);
        // To deserialize we have to call getData because the get method returns a ByteArrayRef
        // in an attempt to improve performance, however at the time of writing this is undocumented.
        byte[] des_pred = predFieldOI.getPrimitiveJavaObject(partialpred).getData(); 
        byte[] des_ci = predFieldOI.getPrimitiveJavaObject(partialci).getData(); 
        byte[] des_betaHat = betaHatFieldOI.getPrimitiveJavaObject(partialBetaHat).getData();
        byte[] des_Ainv = AinvFieldOI.getPrimitiveJavaObject(partialAinv).getData();
        
        HashMap<Double, Double> pred = (HashMap) SerializationUtils.deserialize(des_pred); //I'm not sure this is the correct way to make it into a map....
        HashMap<Double, Double> ci = (HashMap) SerializationUtils.deserialize(des_ci); //I'm not sure this is the correct way to make it into a map....

        myagg.betaHat = (ArrayList) SerializationUtils.deserialize(des_betaHat);

        ArrayList<Double> Ainv = (ArrayList) SerializationUtils.deserialize(des_Ainv);
        myagg.Ainv = Ainv;

        myagg.s = sFieldOI.get(partials);

        if (n == 0) {
          // Just copy the information since there is nothing so far
          myagg.count = m; 
          myagg.pred = pred;
          myagg.ci = ci;
        }

        if (m != 0 && n != 0 && pred != null) {
          // Merge the two partials
          myagg.empty = false;
          myagg.count += m;
          myagg.pred.putAll(pred); //add all key values of one map to the existing map
          myagg.ci.putAll(ci);
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

      // Convert the array list to the DoubleWritable type
      // HashMap<Double, Double> result= myagg.pred;

      // confidence interval
      // rows, cols, and normalization
      long rows = myagg.count;
      int n = myagg.Ainv.size();
      int cols = (int) Math.sqrt(n);
      double degrees_of_freedom = (double)(rows-cols);
      // THIS NEEDS TO BE STUDENT T FOR LOW DEGREES OF FREEDOM
      TDistribution distr = new TDistribution(degrees_of_freedom);
      double qN = distr.inverseCumulativeProbability(0.995); //Two sided 99%

      sb.append("Count: ");
      sb.append(myagg.count);
      // sb.append("\nPredictions: "); //you may not want to see all the predictions; not sure what to do about that. ASK NATE WHAT HE THINKS.
      // sb.append(myagg.pred.toString());

      //output the predictions with the confidence intervals
      sb.append("\n\nError Bounds:\n\n");
      for (Map.Entry<Double, Double> entry : myagg.pred.entrySet()) {
        Double key = entry.getKey();
        Double value = entry.getValue();
        sb.append(key);
        sb.append(" +/- ");
        sb.append(qN*Math.sqrt(myagg.ci.get(key)*myagg.s));
        sb.append("\n");
      }
      sb.append("\n\nwith 99% Confidence");
      result.set(sb.toString());
      return result;

    }

  }
}
