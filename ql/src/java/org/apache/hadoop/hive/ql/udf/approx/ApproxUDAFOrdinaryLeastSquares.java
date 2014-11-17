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
@Description(name = "approx_ols", value = "_FUNC_(x) - Performs an ordinary least squares regression on a set of columns within a tolerence")
public class ApproxUDAFOrdinaryLeastSquares extends AbstractGenericUDAFResolver {

  static final Log LOG = LogFactory.getLog(ApproxUDAFOrdinaryLeastSquares.class.getName());

  @Override
  public GenericUDAFEvaluator getEvaluator(TypeInfo[] parameters)
      throws SemanticException {
    if (parameters.length < 2) {
      throw new UDFArgumentTypeException(parameters.length - 1,
          "Expected at least two arguments but got " + parameters.length);
    }

    for (int i=0; i<parameters.length; i++) {
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
    return new ApproxUDAFOrdinaryLeastSquaresNumeric();
  }

  /**
   * ApproxUDAFOrdinaryLeastSquaresNumeric.
   *
   */
  public static class ApproxUDAFOrdinaryLeastSquaresNumeric extends GenericUDAFEvaluator {
    // For PARTIAL1 and COMPLETE
    private ArrayList<PrimitiveObjectInspector> inputOI;

    // For PARTIAL2 and FINAL
    private StructObjectInspector soi;
    private StructField countField;
    private StructField AField;
    private StructField bField;
    private LongObjectInspector countFieldOI;
    private BinaryObjectInspector AFieldOI;
    private BinaryObjectInspector bFieldOI;

    // For PARTIAL1 and PARTIAL2
    private Object[] partialResult;

    // For FINAL and COMPLETE
    private ArrayList<Double> result;

    @Override
    public ObjectInspector init(Mode m, ObjectInspector[] parameters) throws HiveException {
      assert (parameters.length >= 2);
      super.init(m, parameters);

      // init input
      if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {
        inputOI = new ArrayList<PrimitiveObjectInspector>();
        for (int i=0; i<parameters.length; i++) {
          inputOI.add((PrimitiveObjectInspector) parameters[i]);
        }
      } else {
        soi = (StructObjectInspector) parameters[0];

        countField = soi.getStructFieldRef("count");
        AField = soi.getStructFieldRef("A");
        bField = soi.getStructFieldRef("b");

        countFieldOI = (LongObjectInspector) countField
            .getFieldObjectInspector();
        AFieldOI = (BinaryObjectInspector) AField
            .getFieldObjectInspector();
        bFieldOI = (BinaryObjectInspector) bField
            .getFieldObjectInspector();
      }

      // init output
      if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {
        // The output of a partial aggregation is a struct containing
        // a long count and doubles sum and variance.

        ArrayList<ObjectInspector> foi = new ArrayList<ObjectInspector>();

        foi.add(PrimitiveObjectInspectorFactory.writableBooleanObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableBinaryObjectInspector);
        foi.add(PrimitiveObjectInspectorFactory.writableBinaryObjectInspector);

        ArrayList<String> fname = new ArrayList<String>();
        fname.add("empty");
        fname.add("count");
        fname.add("A");
        fname.add("b");

        partialResult = new Object[4];
        partialResult[0] = new BooleanWritable();
        partialResult[1] = new LongWritable(0);
        partialResult[2] = new BytesWritable();
        partialResult[3] = new BytesWritable();

        return ObjectInspectorFactory.getStandardStructObjectInspector(fname,
            foi);

      } else {
        result = new ArrayList(); 
        return ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
      }

    }

    /** class for storing double sum value. */
    static class SumDoubleAgg implements AggregationBuffer {
      boolean empty;
      long count; // number of elements
      Matrix A;
      Vector b;
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
      myagg.b = null; 
    }

    boolean warned = false;

    @Override
    public void iterate(AggregationBuffer agg, Object[] parameters) throws HiveException {
      assert (parameters.length >= 2);

      Object p = parameters[0];
      if (p != null) {
        int nParams = parameters.length;
        SumDoubleAgg myagg = (SumDoubleAgg) agg;
        try {
          double[] vA = new double[nParams];
          for (int i=0; i<nParams; i++) {
            p = parameters[i];
            PrimitiveObjectInspector thisOI = inputOI.get(i);
            vA[i] = PrimitiveObjectInspectorUtils.getDouble(p, thisOI);
          }
          // Generate the A matrix
          double[][] clipped_mat = new double[1][nParams-1];
          for (int i=0; i<nParams-1; i++)
            clipped_mat[0][i] = vA[i];
          // A = X'*X
          Matrix this_X = new Basic2DMatrix(clipped_mat);
          Matrix this_XT = this_X.transpose();
          Matrix this_A = this_XT.multiply(this_X);
          LOG.warn(this_A.toString());
         
          // Generate the b vector
          Vector this_y = new BasicVector(new double[]{ vA[nParams-1] });
          // b = X'*y
          Vector this_b = this_XT.multiply(this_y);
          
          // We have to test for empty A. This is because during our constructor we 
          // can't access the number of parameters in the query
          if (myagg.A == null) {
            myagg.A = this_A;
          } else {
            myagg.A = myagg.A.add(this_A);
          }
          // Same for b
          if (myagg.b == null) {
            myagg.b = this_b;
          } else {
            myagg.b = myagg.b.add(this_b);
          }

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
    
      // We have to serialize our matricies to bytes. We do this so that we can live on disk between
      // map and reduce phases under the hood (especially if we can't live in memory).      
      byte[] serializedA = SerializationUtils.serialize(myagg.A);
      byte[] serializedb = SerializationUtils.serialize(myagg.b);

      ((BytesWritable) partialResult[2]).set(serializedA, 0, serializedA.length);
      ((BytesWritable) partialResult[3]).set(serializedb, 0, serializedb.length);
      return partialResult;
    }

    @Override
    public void merge(AggregationBuffer agg, Object partial) throws HiveException {

      if (partial != null) {

        SumDoubleAgg myagg = (SumDoubleAgg) agg;
        myagg.empty = false;

        Object partialCount = soi.getStructFieldData(partial, countField);
        Object partialA = soi.getStructFieldData(partial, AField);
        Object partialb = soi.getStructFieldData(partial, bField);

        long n = myagg.count;
        long m = countFieldOI.get(partialCount);
        // To deserialize we have to call getData because the get method returns a ByteArrayRef
        // in an attempt to improve performance, however at the time of writing this is undocumented.
        byte[] des_A = AFieldOI.getPrimitiveJavaObject(partialA).getData();
        byte[] des_b = bFieldOI.getPrimitiveJavaObject(partialb).getData();

        Matrix A = (Basic2DMatrix) SerializationUtils.deserialize(des_A); 
        Vector b = (BasicVector) SerializationUtils.deserialize(des_b);

        if (n == 0) {
          // Just copy the information since there is nothing so far
          myagg.count = m; 
          myagg.A = A;
          myagg.b = b;
        }

        if (m != 0 && n != 0 && A != null && b != null) {
          // Merge the two partials
          myagg.empty = false;
          myagg.count += m;
          myagg.A = myagg.A.add(A);
          myagg.b = myagg.b.add(b);
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

      LeastSquaresSolver solver = new LeastSquaresSolver(myagg.A);
      Vector sparse_solved = solver.solve(myagg.b);
      BasicVector solved = (BasicVector) sparse_solved.copy(LinearAlgebra.BASIC1D_FACTORY);
      double bhat[] = solved.toArray();
      // Convert the array list to the DoubleWritable type
      ArrayList<DoubleWritable> result = new ArrayList<DoubleWritable>();
      for (double d : bhat)
        result.add(new DoubleWritable(d)); 
      
  
      // We'll decompose into QR to check for rank. We require non-perfect 
      // multicollinearity for proper linear least sqaures approximation, aka
      // the columns must not be linearly dependent in any way, aka we require 
      // matrix A to be full rank
      MatrixDecompositor decom = myagg.A.withDecompositor(LinearAlgebra.RAW_QR);
      Matrix[] qrr = decom.decompose();
      Matrix r = qrr[1];
      // If any row is zero we're rank deficient.
      for (int i = 0; i < r.rows(); i++) {
        if (r.get(i, i) == 0.0) {
          throw new UDFArgumentTypeException(0, "The columns you have selected are linearly dependent or close to it. Please use uncorrelated variables for regression");
        } else if (r.get(i,i) <= 0.001) {
          sb.append("Warning: Selected columns are nearly colinear, result may be inaccurate\n");
        }
      }
    

      sb.append("Count: ");
      sb.append(myagg.count);
      sb.append("\nA: ");
      sb.append(myagg.A.toString());
      sb.append("b: ");
      sb.append(myagg.b.toString());
      sb.append("\nCoeffs: ");
      sb.append(solved.toString());

      return result;

    }

  }
}
