package com.stitchfix.mbest

import breeze.linalg.{DenseMatrix, DenseVector}
import com.stitchfix.mbest.FirthLogisticRegression.firthLogisticRegression
import org.scalatest.FunSuite

class FirthLogisticRegressionTest
  extends FunSuite
  with MatrixComparisonTestUtils {

  test("firthLogisticRegression") {
    val X: DenseMatrix[Double] = DenseMatrix(1.0)
    val y_pos: DenseVector[Double] = DenseVector(1.0)
    val y_neg: DenseVector[Double] = DenseVector(0.0)

    val beta_y_pos: DenseVector[Double] = firthLogisticRegression(X, y_pos)
    val beta_y_neg: DenseVector[Double] = firthLogisticRegression(X, y_neg)

    vectorsShouldBeEqual(beta_y_pos, -beta_y_neg)
    vectorsShouldBeEqual(beta_y_pos, DenseVector(1.0986107))
  }

}
