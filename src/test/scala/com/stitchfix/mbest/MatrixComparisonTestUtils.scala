package com.stitchfix.mbest

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.Matchers

trait MatrixComparisonTestUtils extends Matchers {
  def vectorsShouldBeEqual(A: DenseVector[Double], B: DenseVector[Double], threshold: Double = 1E-6) {
    for (i <- 0 until A.length)
      A(i) should be(B(i) +- threshold)
  }

  def matricesShouldBeEqual(A: DenseMatrix[Double], B: DenseMatrix[Double], threshold: Double = 1E-6) {
    for (i <- 0 until A.rows; j <- 0 until A.cols)
      A(i, j) should be(B(i, j) +- threshold)
  }
}
