package com.stitchfix.mbest

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import com.stitchfix.mbest.MatrixUtils._
import org.scalatest.FunSuite

class MatrixUtilsTest
  extends FunSuite
  with MatrixComparisonTestUtils {

  test("compactSVD") {
    val m: DenseMatrix[Double] = DenseMatrix(
      (1.0, 1.0),
      (1.0, 2.0),
      (1.0, 3.0),
      (1.0, 4.0),
      (1.0, 5.0),
      (1.0, 6.0),
      (1.0, 7.0),
      (1.0, 8.0),
      (1.0, 9.0)
    )
    val mm: DenseMatrix[Double] = DenseMatrix.horzcat(m, m)
    val svd_mm: CompactSVD = compactSVD(mm, 2)

    // we should be able to recreate mm from svd_mm
    matricesShouldBeEqual(mm, svd_mm.U * diag(svd_mm.s) * svd_mm.Vt)

    matricesShouldBeEqual(svd_mm.U, DenseMatrix(
      (-0.066963, 0.610978),
      (-0.124743, 0.495530),
      (-0.182523, 0.380083),
      (-0.240304, 0.264635),
      (-0.298084, 0.149188),
      (-0.355864, 0.033741),
      (-0.413644, -0.081707),
      (-0.471425, -0.197154),
      (-0.529205, -0.312602)
    ))

    vectorsShouldBeEqual(svd_mm.s, DenseVector(24.172366, 1.922683))

    matricesShouldBeEqual(svd_mm.Vt, DenseMatrix(
      (-0.110984, -0.698343, -0.110984, -0.698343),
      (0.698343, -0.110984, 0.698343, -0.110984)
    ))

    matricesShouldBeEqual(svd_mm.D, DenseMatrix(
      (24.172366, 0.000000),
      (0.000000, 1.922683)
    ))

    matricesShouldBeEqual(svd_mm.Dneg, DenseMatrix(
      (0.041370, 0.000000),
      (0.000000, 0.520107)
    ))

    matricesShouldBeEqual(svd_mm.Dneg2, DenseMatrix(
      (0.001711, 0.000000),
      (0.000000, 0.270511)
    ))

    matricesShouldBeEqual(svd_mm.UD, DenseMatrix(
      (-1.618654, 1.174716),
      (-3.015339, 0.952748),
      (-4.412025, 0.730779),
      (-5.808710, 0.508810),
      (-7.205395, 0.286841),
      (-8.602081, 0.064872),
      (-9.998766, -0.157096),
      (-11.395451, -0.379065),
      (-12.792137, -0.601034)
    ))

    matricesShouldBeEqual(svd_mm.V1t, DenseMatrix(
      (-0.110984, -0.698343),
      (0.698343, -0.110984)
    ))

    matricesShouldBeEqual(svd_mm.V2t, DenseMatrix(
      (-0.110984, -0.698343),
      (0.698343, -0.110984)
    ))

  }

  test("projectPSD") {
    val m: DenseMatrix[Double] = DenseMatrix(
      (1.0, 2.0, 3.0),
      (2.0, 3.0, 4.0),
      (3.0, 4.0, 5.0)
    )

    matricesShouldBeEqual(projectPSD(m), DenseMatrix(
      (1.427105, 2.073490, 2.719875),
      (2.073490, 3.012645, 3.951800),
      (2.719875, 3.951800, 5.183725)
    ))

  }

  test("sqrtSymMatrix") {
    val m: DenseMatrix[Double] = DenseMatrix(
      (1.427105, 2.073490, 2.719875),
      (2.073490, 3.012645, 3.951800),
      (2.719875, 3.951800, 5.183725)
    )

    matricesShouldBeEqual(sqrtSymMatrix(m), DenseMatrix(
      (0.4600341, 0.6683993, 0.8767646),
      (0.6683993, 0.9711404, 1.2738814),
      (0.8767646, 1.2738814, 1.6709983)
    ))


    // matrices with very small negative eigenvalues should be rounded to 0
    val m2: DenseMatrix[Double] = DenseMatrix(
      (100.0, 0.0,   0.0),
      (  0.0, 1.0,   0.0),
      (  0.0, 0.0, -1e-8)
    )

    matricesShouldBeEqual(sqrtSymMatrix(m2), DenseMatrix(
      (10.0, 0.0, 0.0),
      ( 0.0, 1.0, 0.0),
      ( 0.0, 0.0, 0.0)
    ))

  }

}
