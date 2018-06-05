package com.stitchfix.mbest

import breeze.linalg.eigSym.DenseEigSym
import breeze.linalg.svd.DenseSVD
import breeze.linalg.{DenseMatrix, DenseVector, diag, eigSym, max, min, svd}
import breeze.numerics.{pow, round, sqrt}

/** Utility functions related to matrices. */
object MatrixUtils {

  /** Used when determining which singular values are non-zero  */
  lazy val EPSILON: Double = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

  /** Enforces that the matrix should be symmetric.
    *
    * @param m a matrix
    * @return a matrix
    */
  def removeSymmRoundOffError(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    (m + m.t) / 2.0
  }


  /** Computes a compact singular value decomposition.
    *
    * For the M-by-N matrix it will estimate it's rank K as the number of singular values bigger than zero. An
    * additional input is required to demarcate which columns are associated with fixed effects, and which columns are
    * associated with random effects.
    *
    * - an M-by-K matrix of left vectors,
    * - a K vector of largest singular values in sorted order
    * - a K-by-N matrix of right vectors
    *
    * @param m a matrix
    * @param numFixedEffects an integer stating the number of fixed effects
    * @return a Compact SVD object, including left singular vectors, right singular vectors, non-zero singular values,
    *         and several matrix values necessary for the moment based mixed models algorithm.
    */
  def compactSVD(m: DenseMatrix[Double], numFixedEffects: Int): CompactSVD = {
    val svdA: DenseSVD = svd.reduced(m)

    val eps: Double = EPSILON
    val tolerance: Double = max(svdA.S) * math.max(m.cols, m.rows) * eps

    val firstNonZeroEigenvalue: Int = svdA.S.toArray.indexWhere(_ <= tolerance)
    val restrictedIndex: Range = if (firstNonZeroEigenvalue == -1) {
      svdA.S.toArray.indices
    } else {
      0 until firstNonZeroEigenvalue
    }

    CompactSVD(
      svdA.U(::, restrictedIndex),
      svdA.S(restrictedIndex),
      svdA.Vt(restrictedIndex, ::),
      numFixedEffects
    )
  }

  /** Projects the matrix onto the cone of positive semidefinite matrices.
    *
    * @param m a matrix
    * @return PSD matrix projection
    */
  def projectPSD(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val symm_m: DenseMatrix[Double] = removeSymmRoundOffError(m)
    val symm_m_eig: DenseEigSym = eigSym(symm_m)  // enforces that m is symmetric
    val λ: DenseVector[Double] = symm_m_eig.eigenvalues
    val Q: DenseMatrix[Double] = symm_m_eig.eigenvectors

    val λ_positive: DenseVector[Double] = max(λ, 0.0)
    // As m is symmetric, Q.t = Q**(-1)
    Q * diag(λ_positive) * Q.t
  }

  /** Computes a square root of a symmetric matrix using an eigen decomposition.
    *
    * @param m a matrix
    * @return square root of the matrix
    */
  def sqrtSymMatrix(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val a_eig: DenseEigSym = eigSym(m)  // enforces that m is symmetric
    val λ: DenseVector[Double] = roundDecimalPlace(a_eig.eigenvalues)
    assert(min(λ) >= 0.0)

    val Q: DenseMatrix[Double] = a_eig.eigenvectors
    Q * diag(sqrt(λ)) * Q.t
  }

  /** Rounds a DenseVector[Double] object to a specified decimal place.
    *
    * @param v a vector
    * @param digits the level of decimals places to round to
    * @return a rounded vector
    */
  def roundDecimalPlace(v: DenseVector[Double], digits: Int = 7): DenseVector[Double] = {
    val inflationFactor: Double = pow(10, digits)
    val inflatedVector: DenseVector[Double] = round(v * inflationFactor).mapValues(_.toDouble)
    inflatedVector / inflationFactor
  }

  /** A compact singular value decomposition class, including lazy evaluations of several derived terms.
    *
    * All zero values singular values are dropped, along with the corresponding singular values.
    *
    * @param U a DenseMatrix, the left singular values
    * @param s a DenseVector, the singular values
    * @param Vt a DenseMatrix, the right singular values
    * @param numFixedEffects an integer, the number of fixed effects coefficients.
    */
  case class CompactSVD(
    U: DenseMatrix[Double],
    s: DenseVector[Double],
    Vt: DenseMatrix[Double],
    numFixedEffects: Int
  ) {
    @transient lazy val D: DenseMatrix[Double] = diag(s)
    @transient lazy val Dneg: DenseMatrix[Double] = diag(pow(s, -1))
    @transient lazy val Dneg2: DenseMatrix[Double] = diag(pow(s, -2))
    @transient lazy val UD: DenseMatrix[Double] = U * D
    @transient lazy val V1t: DenseMatrix[Double] = Vt(::, 0 until numFixedEffects)
    @transient lazy val V2t: DenseMatrix[Double] = Vt(::, numFixedEffects until Vt.cols)

    def rank: Int = s.length
  }

}
