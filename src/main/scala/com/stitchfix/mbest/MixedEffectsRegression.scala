/*
 * Copyright 2011 Stitch Fix
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.stitchfix.mbest

import breeze.linalg.{DenseMatrix, DenseVector, diag, kron, pinv}
import breeze.numerics.exp
import com.stitchfix.mbest.FirthLogisticRegression.firthLogisticRegression
import com.stitchfix.mbest.MatrixUtils.{CompactSVD, compactSVD, projectPSD, removeSymmRoundOffError, sqrtSymMatrix}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Row}

// TODO: transform log-odds into either probabilities or classes
// TODO: for binomial require two classes


/**
  * Fits a hierarchical generalized linear model using moment-based estimation.
  *
  * References:
  *   Perry, P. O. (2015) "Fast Moment-Based Estimation for Hierarchical Models", in https://arxiv.org/abs/1504.04941
  *
  * @param uid a string,
  */
class MixedEffectsRegression (
  override val uid: String
) extends Estimator[MixedEffectsRegressionModel]
  with MixedEffectsRegressionBase {

  def this() = this(Identifiable.randomUID("MixedEffectsRegression"))

  override def copy(extra: ParamMap): MixedEffectsRegression = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = true)
  }

  override def fit(rawDataset: Dataset[_]): MixedEffectsRegressionModel = {
    transformSchema(rawDataset.schema, logging = true)
    val dataset: Dataset[_] = transformDataset(rawDataset)

    val numFixedEffects: Int = if ($(fixedEffectInterceptParam)) {
      $(fixedEffectColsParam).length + 1
    } else {
      $(fixedEffectColsParam).length
    }

    val numRandomEffects: Int = if ($(randomEffectInterceptParam)) {
      $(randomEffectColsParam).length + 1
    } else {
      $(randomEffectColsParam).length
    }

    val grouped_data: RDD[(String, MixedEffectsRegression.Data)] = MixedEffectsRegression.generateGroupedData(
      dataset, $(groupColPram), $(responseColPram)
    )

    val grouped_svd: RDD[(String, (MixedEffectsRegression.Data, CompactSVD))] = {
      MixedEffectsRegression.generateGroupedCompactSVD(grouped_data, numFixedEffects)
    }
    val grouped_η0: RDD[(String, (MixedEffectsRegression.Data, CompactSVD, MixedEffectsRegression.Estimate_η0))] = {
      MixedEffectsRegression.estimateGrouped_η0(grouped_svd, $(familyParam))
    }

    val φ: Double = MixedEffectsRegression.estimate_φ($(familyParam), grouped_η0)
    val φ_bcast: Broadcast[Double] = dataset.rdd.context.broadcast(φ)

    val β: DenseVector[Double] = MixedEffectsRegression.estimate_β(grouped_η0)
    val β_bcast: Broadcast[DenseVector[Double]] = dataset.rdd.context.broadcast(β)

    val Σ: DenseMatrix[Double] = {
      MixedEffectsRegression.estimate_Σ(grouped_η0, β_bcast, φ_bcast, numFixedEffects, numRandomEffects)
    }
    val Σ_sqrt: DenseMatrix[Double] = sqrtSymMatrix(Σ)
    val Σ_sqrt_bcast: Broadcast[DenseMatrix[Double]] = dataset.rdd.context.broadcast(Σ_sqrt)

    // TODO: Make randomEffects a DataFrame?
    val randomEffects: Map[String, DenseVector[Double]] = MixedEffectsRegression.estimateRandomEffects(
      grouped_η0, β_bcast, φ_bcast, Σ_sqrt_bcast
    ).map(identity) // To make random effects map of vectors serializable

    val model: MixedEffectsRegressionModel = new MixedEffectsRegressionModel(
      uid, numFixedEffects, numRandomEffects, $(familyParam), β, φ, Σ, randomEffects
    )

    model
      .setFamilyParam($(familyParam))
      .setResponseCol($(responseColPram))
      .setPredictionCol($(predictionColPram))
      .setGroupCol($(groupColPram))
      .setFixedEffectCols($(fixedEffectColsParam))
      .setRandomEffectCols($(randomEffectColsParam))
      .setFixedEffectInterceptParam($(fixedEffectInterceptParam))
      .setRandomEffectInterceptParam($(randomEffectInterceptParam))

  }

}

/** companion object */
object MixedEffectsRegression
{

   /** TODO: describe what is actually going on here
    *
    * @return Instance
    */
  def generateGroupedData(dataset: Dataset[_], groupCol: String, responseCol: String): RDD[(String, Data)] = {

    val instances: RDD[Instance] = dataset.select(
      col(groupCol),
      col(responseCol),
      col("fixedEffects"),
      col("randomEffects")
    ).rdd.map {
      case Row(
        group: String,
        response: Double,
        fixedEffects: Vector,
        randomEffects: Vector
      ) => Instance(group, response, fixedEffects, randomEffects)
    }

    // Organizing <X>, <Z> and <y> for each group
    val groupedInstances: RDD[(String, Iterable[Instance])] = instances.groupBy(_.group)

    groupedInstances.map {
      case (group: String, groupInstances: Iterable[Instance]) =>
        // Materializing instances and fixing an iteration order
        val groupInstancesList: List[Instance] = groupInstances.toList

        // TODO: is :_* operator too slow?
        (
          group,
          Data(
            X = DenseMatrix(groupInstancesList.map(_.fixedEffects.toArray):_*),
            Z = DenseMatrix(groupInstancesList.map(_.randomEffects.toArray):_*),
            y = DenseVector(groupInstancesList.map(_.response).toArray)
          )
        )
    }

  }

  /** Compute a compact singular value decomposition for each grouping level.
    *
    * Compute a compact, singular value decomposition on the column-wise concatenated fixed- and random-effects design
    * matrices. See section 3.2 of Perry, P. O. (2015).
    *
    * @param grouped_data, an RDD of string and `Data` tuples
    * @param numFixedEffects, an integer, the number of fixed effects columns
    * @return an RDD object of string, Data, and CompactSVD tuples
    */
  def generateGroupedCompactSVD(grouped_data: RDD[(String, Data)], numFixedEffects: Int):
      RDD[(String, (Data, CompactSVD))] = {

    grouped_data.map {
      case (group, data) => (group, (data, compactSVD(data.F, numFixedEffects)))
    }
  }

  /** Estimate generalized linear model coefficients for each grouping level
    *
    * For each grouping level, estimates a scaled, concatenated coefficients vector, derived from first the (global)
    * fixed effects coefficients, then the (local) random effects coefficeints. See section 3.2 of Perry, P. O. (2015).
    *
    * @param grouped_svd, an RDD of string, Data, and CompactSVD tuples
    * @param family, a string, delineating the type of generalized linear model. Currently support 'gaussian' and
    *              'binomial'
    * @return an RDD object of string, Data, CompactSVD, and Estimate_η0 tuples
    */
  def estimateGrouped_η0(grouped_svd: RDD[(String, (Data, CompactSVD))], family: String):
      RDD[(String, (Data, CompactSVD, Estimate_η0))] = {

    grouped_svd.map {
      case (group, (data, svd)) => (group, (data, svd, estimate_η0(group, family, data, svd)))
    }
  }

  /** Given the type of family of generalized linear models, estimates η0 using the appropriate method.
    *
    * @param group A string, the grouping level id
    * @param family, a string, delineating the type of generalized linear model. Currently support 'gaussian' and
    *              'binomial'
    * @param data a Data package case class
    * @param svd a compact singular value object
    * @return an Estimate_η0 package case class
    */
  def estimate_η0(group: String, family: String, data: Data, svd: CompactSVD): Estimate_η0 = {
    family match {
      case "gaussian" => estimateGaussian_η0(data, svd)
      case "binomial" => estimateBinomial_η0(data, svd)
      case _ => throw new IllegalArgumentException(s"Family '$family' is not supported")
    }
  }

  /** Estimates η0 for gaussian type generalized linear models
    *
    * @param data a Data package case class
    * @param svd a compact singular value object
    * @return an Estimate_η0 package case class
    */
  def estimateGaussian_η0(data: Data, svd: CompactSVD): Estimate_η0 = {
    // Regressing y ~ UD * η0
    val η0_μ: DenseVector[Double] = svd.Dneg * svd.U.t * data.y // η0_hat ~ Vt * η_hat
    val η0_Σ: DenseMatrix[Double] = svd.Dneg2

    Estimate_η0(η0_μ, η0_Σ)
  }

  /** Estimates η0 for binomial type generalized linear models
    *
    * See step (a) in Section 3.4 of Perry, P. O. (2015).
    *
    * @param data a Data package case class
    * @param svd a compact singular value object
    * @return an Estimate_η0 package case class
    */
  def estimateBinomial_η0(data: Data, svd: CompactSVD): Estimate_η0 = {
    val η0: DenseVector[Double] = firthLogisticRegression(svd.UD, data.y)

    val logOdds: DenseVector[Double] = svd.UD * η0
    val prob: DenseVector[Double] = 1.0 / (exp(-logOdds) + 1.0)
    val W: DenseMatrix[Double] = diag(prob *:* (1.0 - prob))
    val η0_Σ: DenseMatrix[Double] = removeSymmRoundOffError(pinv(svd.UD.t * W * svd.UD))

    Estimate_η0(η0, η0_Σ)
  }

  /** Given the type of family of generalized linear models, estimates the dispersion parameter φ.
    *
    * @param family, a string, delineating the type of generalized linear model. Currently support 'gaussian' and
    *              'binomial'
    * @param grouped_η0 an RDD containing the η0 estimates.
    * @return a double, the models estimated dispersion φ.
    */
  def estimate_φ(family: String, grouped_η0: RDD[(String, (Data, CompactSVD, Estimate_η0))]): Double = {

    family match {
      case "gaussian" => estimateGaussian_φ(grouped_η0)
      case "binomial" => 1.0
      case _ => throw new IllegalArgumentException(s"Family '$family' is not supported")
    }

  }

  /** Estimate Dispersion for gaussian family generalized linear model.
    *
    * See step (b) in Section 3.4 of Perry, P. O. (2015).
    *
    * @param grouped_η0 an RDD containing the η0 estimates.
    * @return a double, the models estimated dispersion φ.
    */
  def estimateGaussian_φ(grouped_η0: RDD[(String, (Data, CompactSVD, Estimate_η0))]): Double = {
    val groups_φ: RDD[(String, (Data, CompactSVD, Estimate_η0, Estimate_φ))] =grouped_η0.map {
      case (group, (data, svd, η0_est)) =>
        (group, (data, svd, η0_est, estimateGaussianGroupTerm_φ(data, svd, η0_est)))
    }

    groups_φ.values.map(x => x._4.φ_mult_weight).sum / groups_φ.values.map(x => x._4.φ_weight).sum
  }

  /** Estimates the contribution to dispersion from a specific group.
    *
    * Model dispersion, φ = Σ_i φ_mult_weight_i / φ_weight_i. See step (b) in Section 3.4 of Perry, P. O. (2015).
    *
    * @param data a Data package case class
    * @param svd a compact singular value object
    * @param η0_est an Estimate_η0 object
    * @return and an Estimate_φ object
    */
  def estimateGaussianGroupTerm_φ(data: Data, svd: CompactSVD, η0_est: Estimate_η0): Estimate_φ = {
    val eps: DenseVector[Double] = data.y - svd.UD * η0_est.η0_μ
    val φ_mult_weight: Double = eps.t * eps
    val φ_weight: Int = data.F.rows - svd.rank

    Estimate_φ(φ_mult_weight, φ_weight)
  }

  /** Estimate fixed effects vector.
    *
    * See (3) and (4) in Perry, P. O. (2015). Using a tree reduce to avoid making the driver a bottleneck.
    *
    * @param grouped_η0 an RDD object, containing design matrices, compact SVDs, and estimated η0 vectors.
    * @return DenseVector[Double], the estimated fixed effects vector
    */
  def estimate_β(grouped_η0: RDD[(String, (Data, CompactSVD, Estimate_η0))]): DenseVector[Double] = {

    val grouped_β: RDD[(String, Estimate_β)] = grouped_η0.map {
      case (group, (_, svd, η0_est)) => (group, estimateGroupTerm_β(svd, η0_est))
    }
    val agg_β = grouped_β.map(_._2).treeReduce((a, b) => Estimate_β(a.Ω + b.Ω, a.V1_η0 + b.V1_η0))
    pinv(agg_β.Ω) * agg_β.V1_η0 // TODO: make this more numerically stable
  }

  /** Estimates group components used for estimation of fixed effects vector
    *
    * See (3) and (4) in Perry, P. O. (2015).
    *
    * @param svd a compact singular value object
    * @param η0_est an Estimate_η0 object
    * @return an Estimate_β case class
    */
  def estimateGroupTerm_β(svd: CompactSVD, η0_est: Estimate_η0): Estimate_β = {
    val V1_V1T: DenseMatrix[Double] = svd.V1t.t * svd.V1t
    val V1_η0: DenseVector[Double] = svd.V1t.t * η0_est.η0_μ  // V1 * η0_hat ~ V1 * Vt * η_hat

    Estimate_β(V1_V1T, V1_η0)
  }

  /** Estimate the covariance matrix of the prior distribution of random effects vectors
    *
    * Corresponds to (5) and (6) in Perry, P. O. (2015). Note, some notational differences exist between this package and
    * the original paper.
    *
    * @param grouped_η0 an RDD object, containing design matrices, compact SVDs, and estimated η0 vectors.
    * @param β_bcast a broadcast DenseVector, the fixed effects estimate
    * @param φ_bcast a broadcast Double, the model dispersion estimate
    * @param numFixedEffects an integer, the number of fixed effects parameters
    * @param numRandomEffects an integer, the number of random effects parameters
    * @return A DenseMatrix[Double], the random effects covariance matrix
    */
  def estimate_Σ(grouped_η0: RDD[(String, (Data, CompactSVD, Estimate_η0))],
                 β_bcast: Broadcast[DenseVector[Double]], φ_bcast: Broadcast[Double],
                 numFixedEffects: Int, numRandomEffects: Int): DenseMatrix[Double]= {
    val grouped_Σ: RDD[(String, Estimate_Σ)] = grouped_η0.map {
      case (group, (_, svd, η0_est)) => (group, estimateGroupTerm_Σ(svd, η0_est, β_bcast.value))
    }

    val agg_Σ = grouped_Σ.map(_._2).treeReduce(
      (a, b) => Estimate_Σ(a.Ω2 + b.Ω2, a.A_β + b.A_β, a.B + b.B)
    )
    val M: DenseMatrix[Double] = agg_Σ.A_β - φ_bcast.value * agg_Σ.B
    val M_vec: DenseVector[Double] = M.flatten()
    val Σ_vec: DenseVector[Double] = pinv(agg_Σ.Ω2) * M_vec
    val Σ_momentEstimate: DenseMatrix[Double] = Σ_vec.toDenseMatrix.reshape(numRandomEffects, numRandomEffects)
    removeSymmRoundOffError(projectPSD(Σ_momentEstimate))
  }

  /** Estimates specified group components required to later estimate random effects covariance matrix.
    *
    * Corresponds to (5) and (6) in Perry, P. O. (2015). Note, some notational differences exist between this package
    * and the original paper.
    *
    * @param svd a compact singular value object
    * @param η0_est an Estimate_η0 object
    * @param β a DenseVector, the fixed effects estimate
    * @return an Estimate_Σ case class.
    */
  def estimateGroupTerm_Σ(svd: CompactSVD, η0_est: Estimate_η0, β: DenseVector[Double]): Estimate_Σ = {
    val V2_V2T: DenseMatrix[Double] = svd.V2t.t * svd.V2t
    val V2_V2T_kron_V2_V2T: DenseMatrix[Double] = kron(V2_V2T, V2_V2T)

    val V2_η0_minus_V1t_β: DenseVector[Double] = svd.V2t.t * (η0_est.η0_μ - svd.V1t * β)
    val A_β_elem: DenseMatrix[Double] = V2_η0_minus_V1t_β * V2_η0_minus_V1t_β.t

    val B_unnormed_elem: DenseMatrix[Double] = svd.V2t.t * svd.Dneg2 * svd.V2t

    Estimate_Σ(V2_V2T_kron_V2_V2T, A_β_elem, B_unnormed_elem)
  }

  /** Perform empirical-Bayes mean estimation of random effects vectors, across all groups.
    *
    * Corresponds to step (f) in Section 3.4 of Perry, P. O. (2015).
    *
    * @param grouped_η0 an RDD object, containing design matrices, compact SVDs, and estimated η0 vectors.
    * @param β_bcast a broadcast DenseVector, the fixed effects estimate
    * @param φ_bcast a broadcast Double, the model dispersion estimate
    * @param Σ_sqrt_bcast a broadcast DenseMatrix, the square root of the random effects covariance matrix.
    * @return a Map object of the group level name and the random effect vector.
    */
  def estimateRandomEffects(grouped_η0: RDD[(String, (Data, CompactSVD, Estimate_η0))],
                            β_bcast: Broadcast[DenseVector[Double]],
                            φ_bcast: Broadcast[Double],
                            Σ_sqrt_bcast: Broadcast[DenseMatrix[Double]]): Map[String, DenseVector[Double]] = {
    // Estimating <u> for each group
    // Corresponds to [1] ch. 3.4 estimation procedure, cl. (f)
    val grouped_u: RDD[(String, DenseVector[Double])] = grouped_η0.map {
      case (group, (_, svd, η0_est)) => (
        group, estimateRandomEffectGroupTerm(svd, η0_est, β_bcast.value, Σ_sqrt_bcast.value, φ_bcast.value)
      )
    }

    grouped_u.collect().toMap
  }

  /** Perform empirical-Bayes mean estimation of random effects vectors, across a single group.
    *
    * Corresponds to step (f) in Section 3.4 of Perry, P. O. (2015).
    *
    * @param svd a compact singular value object
    * @param η0_est an Estimate_η0 object
    * @param β a DenseVector, the fixed effects estimate
    * @param Σ_sqrt a DenseMatrix, the square root of the random effects covariance matrix.
    * @param φ a Double, the model dispersion estimate
    * @return a DenseVector, the random effects estimate.
    */
  def estimateRandomEffectGroupTerm(svd: CompactSVD, η0_est: Estimate_η0, β: DenseVector[Double],
                                    Σ_sqrt: DenseMatrix[Double], φ: Double): DenseVector[Double] = {
    val v: DenseVector[Double] = η0_est.η0_μ - svd.V1t * β
    val V2t_pinv: DenseMatrix[Double] = pinv(svd.V2t)
    val μ_ml: DenseVector[Double] = V2t_pinv * v
    val Σ_ml: DenseMatrix[Double] = φ * V2t_pinv * η0_est.η0_Σ * V2t_pinv.t
    val Λ_ml: DenseMatrix[Double] = pinv(Σ_ml)
    val I: DenseMatrix[Double] = DenseMatrix.eye[Double](Σ_ml.rows)

    val mat_sum: DenseMatrix[Double] = I + Σ_sqrt * Λ_ml *  Σ_sqrt
    Σ_sqrt * pinv(mat_sum) * Σ_sqrt * Λ_ml * μ_ml
  }

  /** Initial processing unit of data for fitting using moment-based estimation.
    *
    * @param group a string, the name of the grouping level column
    * @param response a vector of doubles, the response values
    * @param fixedEffects a vector, the fixed effects features
    * @param randomEffects a vector, the random effects features
    */
  case class Instance(
    group: String,
    response: Double,
    fixedEffects: Vector,
    randomEffects: Vector
  )

  /** Basic unit of training data used for each group level.
    *
    * Used for fitting models with structure as in (1) from Perry, P. O. (2015)
    *
    * @param X a matrix, the fixed effects design matrix
    * @param Z a matrix, the random effects design matrix.
    * @param y a vector, the response (outcome) vector
    */
  case class Data(
    X: DenseMatrix[Double],
    Z: DenseMatrix[Double],
    y: DenseVector[Double]
  ) {
    @transient lazy val F: DenseMatrix[Double] = DenseMatrix.horzcat(X, Z)
  }

  /** The point estimate and covariance estimate for η0 for each group level.
    *
    * η0 is a transformation of the concatenation of the fixed and random effects coefficients (for a particular group
    * level). See (2a) and (2b) in Perry, P. O. (2015). Note, that our \hat{η0}_i is the same as V_iT\hat{η}_i.
    *
    * @param η0_μ a vector, the point estimate of η0
    * @param η0_Σ a matrix, the covariance of η0
    */
  case class Estimate_η0(
    η0_μ: DenseVector[Double],
    η0_Σ: DenseMatrix[Double]
  )

  /** The group level dispersion estimate and associated weight.
    *
    * Used for computing step (b) in Section 3.4 of Perry, P. O. (2015).
    *
    * @param φ_mult_weight a Double
    * @param φ_weight an Integer
    */
  case class Estimate_φ(
    φ_mult_weight: Double,
    φ_weight: Int
  )

  /** Intermediary group level terms used to estimate β.
    *
    * See (3) and (4) in Perry, P. O. (2015).
    *
    * @param Ω a matrix
    * @param V1_η0 a vector, a weighted combination of η0
    */
  case class Estimate_β(
    Ω: DenseMatrix[Double],
    V1_η0: DenseVector[Double]
  )

  /** Intermediary group level terms used to estimate Σ.
    *
    * See (5) and (6) in Perry, P. O. (2015). Note there are some syntactic differences between variables names in this
    * package and the original paper.
    *
    * @param Ω2 a matrix
    * @param A_β a matrix
    * @param B a matrix
    */
  case class Estimate_Σ(
    Ω2: DenseMatrix[Double],
    A_β: DenseMatrix[Double],
    B: DenseMatrix[Double]
  )

}
