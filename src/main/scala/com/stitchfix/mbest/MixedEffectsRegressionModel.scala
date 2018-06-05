package com.stitchfix.mbest

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Model fitted by MixedEffectsRegression.
  *
  * @param numFixedEffects - the number of fixed effects
  * @param numRandomEffects - the number of random effects
  * @param β - estimated fixed effects vector
  * @param Σ - estimated random effects covariance matrix
  * @param randomEffects - estimated random effects vectors, one per each group
  */
class MixedEffectsRegressionModel(
  override val uid: String,
  val numFixedEffects: Int,
  val numRandomEffects: Int,
  val family: String,
  val β: DenseVector[Double],
  val φ: Double,
  val Σ: DenseMatrix[Double],
  val randomEffects: Map[String, DenseVector[Double]]
) extends Model[MixedEffectsRegressionModel]
  with MixedEffectsRegressionBase {

  override def copy(extra: ParamMap): MixedEffectsRegressionModel = {
    val newModel = copyValues(
      new MixedEffectsRegressionModel(uid, numFixedEffects, numRandomEffects, family, β, φ, Σ, randomEffects),
      extra
    )
    newModel.setParent(parent)
  }

  override def transform(rawDataset: Dataset[_]): DataFrame = {
    transformSchema(rawDataset.schema, logging = true)
    val dataset: Dataset[_] = transformDataset(rawDataset)

    val predict = udf {
      (group: String, fixedEffectsVector: Vector, randomEffectsVector: Vector) =>
        val u = randomEffects.getOrElse(group, DenseVector.zeros[Double](numRandomEffects))

        val x: DenseVector[Double] = DenseVector(fixedEffectsVector.toDense.values)
        val z: DenseVector[Double] = DenseVector(randomEffectsVector.toDense.values)
        require(x.length == numFixedEffects)
        require(z.length == numRandomEffects)

        x.t * β + z.t * u
    }

    dataset.withColumn(
      $(predictionColPram),
      predict(
        col($(groupColPram)),
        col("fixedEffects"),
        col("randomEffects")
      )
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = false)
  }

}
