package com.stitchfix.mbest

import breeze.linalg.{DenseMatrix, DenseVector}
import com.stitchfix.mbest.MatrixUtils.CompactSVD
import com.stitchfix.mbest.MixedEffectsRegression._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.scalatest.{FunSuite, Matchers}

import scala.collection.immutable.Seq

class MixedEffectsRegressionTest
  extends FunSuite
  with Matchers
  with MatrixComparisonTestUtils
  with LocalSpark
  with SleepStudyTestData {

  test("estimateGaussian_η0") {

    val result: Estimate_η0 = estimateGaussian_η0(data308, csvd)
    vectorsShouldBeEqual(result.η0_μ, η0_308.η0_μ, 4)
    matricesShouldBeEqual(result.η0_Σ, η0_308.η0_Σ)

  }

  test("estimateBinomial_η0") {
    val y: DenseVector[Double] = DenseVector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    val data: Data = Data(m, m, y)
    val result: Estimate_η0 = estimateBinomial_η0(data, csvd)
    vectorsShouldBeEqual(result.η0_μ, DenseVector(-0.138339, -2.282826))
    matricesShouldBeEqual(result.η0_Σ, DenseMatrix(
      (0.015789, 0.050418),
      (0.050418, 2.149605)
    ))
  }

  test("estimateGaussian_φ") {
    val testRDD: RDD[(String, (Data, CompactSVD, Estimate_η0))] = localSpark.sparkContext.parallelize(sleepStudyInputData)
    estimateGaussian_φ(testRDD) should be(654.9410 +- 1e-4)
  }

  test("estimate_β") {
    val testRDD: RDD[(String, (Data, CompactSVD, Estimate_η0))] = localSpark.sparkContext.parallelize(sleepStudyInputData)
    vectorsShouldBeEqual(estimate_β(testRDD), DenseVector(251.40510, 10.46729), 6)
  }

  test("estimate_Σ") {
    val testRDD: RDD[(String, (Data, CompactSVD, Estimate_η0))] = localSpark.sparkContext.parallelize(sleepStudyInputData)

    val β: DenseVector[Double] = DenseVector(251.40510, 10.46729)
    val β_bcast: Broadcast[DenseVector[Double]] = testRDD.context.broadcast(β)

    val φ: Double = 654.941
    val φ_bcast: Broadcast[Double] = testRDD.context.broadcast(φ)

    matricesShouldBeEqual(
      estimate_Σ(testRDD, β_bcast, φ_bcast, 2, 2),
      DenseMatrix(
        (565.5154, 11.0554),
        ( 11.0554, 32.6822)
      ), 4)
  }

  test("estimateRandomEffectGroupTerm") {
    val β: DenseVector[Double] = DenseVector(251.40510, 10.46729)
    val Σ_sqrt: DenseMatrix[Double] = DenseMatrix(
      (23.7776103, 0.3749874),
      (0.37498740, 5.7045229)
    )
    val φ: Double = 654.941

    vectorsShouldBeEqual(
      estimateRandomEffectGroupTerm(csvd, η0_308, β, Σ_sqrt, φ),
      DenseVector(2.815664, 9.075540)
    )
  }


  test("fit binomial model") {

    val customSchema = StructType(Array(
      StructField("x1", DoubleType, true),
      StructField("x2", DoubleType, true),
      StructField("y", DoubleType, true),
      StructField("group_id", StringType, true)))

    val rawDat = localSpark.read.format("csv").option("header", "true").schema(customSchema).load("data/binomial_sim.csv")

    val mhglm = {
      new MixedEffectsRegression()
        .setResponseCol("y")
        .setFixedEffectCols(Seq("x1", "x2"))
        .setRandomEffectCols(Seq("x1"))
        .setFamilyParam("binomial")
        .setGroupCol("group_id")
    }

    val model = mhglm.fit(rawDat)

    // parameters should be passed down from the mhglm object to the model object
    assert(model.getFamilyParam == mhglm.getFamilyParam)
    assert(model.getResponseCol == mhglm.getResponseCol)
    assert(model.getPredictionCol == mhglm.getPredictionCol)
    assert(model.getGroupCol == mhglm.getGroupCol)
    assert(model.getFixedEffectCols == mhglm.getFixedEffectCols)
    assert(model.getRandomEffectCols == mhglm.getRandomEffectCols)
    assert(model.getFixedEffectInterceptParam == mhglm.getFixedEffectInterceptParam)
    assert(model.getRandomEffectInterceptParam == mhglm.getRandomEffectInterceptParam)

    val predictions = model.transform(rawDat)
  }

}
