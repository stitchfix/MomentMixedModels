package com.stitchfix.mbest

import breeze.linalg.{DenseMatrix, DenseVector, convert, det, diag, max, min, pinv, rank, sum, trace}
import breeze.numerics.{exp, log}
import breeze.optimize.{DiffFunction, LBFGS}

import scala.collection.immutable.Seq
import scala.math.{min => scalarMin}

/** Firth Logistic Regression */
object FirthLogisticRegression {

  /** Fits a Firth Logistic Regression Model.
    *
    * Firth Logistic Regression uses Jeffreys prior to "regularize" its estimates. This solves separability concerns,
    * as well as reduces small sample bias in logistic regression. In this implimentation we use a LBFGS solver.
    *
    * See:
    * [1] Firth (1993). Bias reduction of maximum likelihood estimates. Biometrika 80, 27–38
    * [2] Kosmidis and Firth (2009). Bias reduction in exponential family nonlinear models. Biometrika 96, 793–804.
    *
    * @param X, a DenseMatrix, the design matrix.
    * @param y, a DenseVector, the response vector. Should only have values 0 and 1.
    * @return a Dense Vector, the estimates coefficients.
    */
  def firthLogisticRegression(X: DenseMatrix[Double], y: DenseVector[Double]): DenseVector[Double] = {
    assert(min(y) >= 0.0)
    assert(max(y) <= 1.0)
    assert(rank(X) == scalarMin(X.rows, X.cols))

    val y_as_double: DenseVector[Double] = convert(y, Double)

    val f = new DiffFunction[DenseVector[Double]] {
      override def calculate(β: DenseVector[Double]): (Double, DenseVector[Double]) = {
        binomialFirthObjective(β, X, y_as_double) -> binomialFirthGradient(β, X, y_as_double)
      }
    }

    // LBFGS parameters copied from scipy.optimize.fmin_l_bfgs_b
    val optimizer = new LBFGS[DenseVector[Double]](maxIter = 15000, m = 7, tolerance = 1E-5)
    val init_β: DenseVector[Double] = DenseVector.zeros[Double](X.cols)

    optimizer.minimize(f, init_β)
  }

  private def binomialFirthObjective(β: DenseVector[Double], X: DenseMatrix[Double], y: DenseVector[Double]): Double = {
    val X_β: DenseVector[Double] = X * β
    val prob: DenseVector[Double] = 1.0 / (exp(-X_β) + 1.0)
    val W: DenseMatrix[Double] = diag(prob *:* (1.0 - prob))

    val log_likelihood: Double = sum(y *:* X_β - log(exp(X_β) + 1.0))
    val log_prior: Double = 0.5 * log(det(X.t * W * X))

    -(log_likelihood + log_prior)
  }

  private def binomialFirthGradient(β: DenseVector[Double], X: DenseMatrix[Double], y: DenseVector[Double]): DenseVector[Double] = {
    val X_β: DenseVector[Double] = X * β
    val prob: DenseVector[Double] = 1.0 / (exp(-X_β) + 1.0)
    val W: DenseMatrix[Double] = diag(prob *:* (1.0 - prob))

    val grad_log_likelihood: DenseVector[Double] = X.t * (y - prob)

    val X_XtWXinv_Xt: DenseMatrix[Double] = X * pinv(X.t * W * X) * X.t
    val grad_prior_components: Seq[Double] = for (i <- 0 until X.cols) yield {
      val X_i: DenseVector[Double] = X(::, i)
      val dW_i: DenseVector[Double] = prob *:* (prob * 2.0 - 1.0) *:* (prob - 1.0)
      val grad_i = 0.5 * trace(X_XtWXinv_Xt * diag(X_i *:* dW_i))
      grad_i
    }
    val grad_log_prior: DenseVector[Double] = DenseVector[Double](grad_prior_components:_*)

    -(grad_log_likelihood + grad_log_prior)
  }

}
