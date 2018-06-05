package com.stitchfix.mbest

import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf


trait MixedEffectsRegressionBase
  extends Params
    with Logging {

  final val familyParam: Param[String] = new Param[String](this, "familyParam", "target family: gaussian/binomial")
  final val responseColPram: Param[String] = new Param[String](this, "responseCol", "response column name")
  final val predictionColPram: Param[String] = new Param[String](this, "predictionCol", "prediction column name")
  final val groupColPram: Param[String] = new Param[String](this, "groupCol", "group column name")
  final val fixedEffectColsParam: Param[Seq[String]] = new Param[Seq[String]](this, "fixedEffectCols", "fixed effects column name")
  final val randomEffectColsParam: Param[Seq[String]] = new Param[Seq[String]](this, "randomEffectCols", "random effects column name")
  final val fixedEffectInterceptParam: Param[Boolean] = new Param[Boolean](this, "fixedEffectInterceptParam", "include an intercept for fixed effects?")
  final val randomEffectInterceptParam: Param[Boolean] = new Param[Boolean](this, "randomEffectInterceptParam", "include an intercept for random effects?")

  setDefault(familyParam, "")
  setDefault(responseColPram, "")
  setDefault(predictionColPram, "mghlm_prediction")
  setDefault(groupColPram, "")
  setDefault(fixedEffectColsParam, Seq())
  setDefault(randomEffectColsParam, Seq())
  setDefault(fixedEffectInterceptParam, true)
  setDefault(randomEffectInterceptParam, true)

  final def setFamilyParam(value: String): this.type = set(familyParam, value)
  final def setResponseCol(value: String): this.type = set(responseColPram, value)
  final def setPredictionCol(value: String): this.type = set(predictionColPram, value)
  final def setGroupCol(value: String): this.type = set(groupColPram, value)
  final def setFixedEffectCols(value: Seq[String]): this.type = set(fixedEffectColsParam, value)
  final def setRandomEffectCols(value: Seq[String]): this.type = set(randomEffectColsParam, value)
  final def setFixedEffectInterceptParam(value: Boolean): this.type = set(fixedEffectInterceptParam, value)
  final def setRandomEffectInterceptParam(value: Boolean): this.type = set(randomEffectInterceptParam, value)

  final def getFamilyParam: String = $(familyParam)
  final def getResponseCol: String = $(responseColPram)
  final def getPredictionCol: String = $(predictionColPram)
  final def getGroupCol: String = $(groupColPram)
  final def getFixedEffectCols: Seq[String] = $(fixedEffectColsParam)
  final def getRandomEffectCols: Seq[String] = $(randomEffectColsParam)
  final def getFixedEffectInterceptParam: Boolean = $(fixedEffectInterceptParam)
  final def getRandomEffectInterceptParam: Boolean = $(randomEffectInterceptParam)

  def validateAndTransformSchema(schema: StructType, fitting: Boolean): StructType = {
    // Checking response column type if training
    if (fitting) {
      val col = $(responseColPram)
      val actualDataType = schema(col).dataType
      require(
        actualDataType == DoubleType,
        s"Column $col must be of type DoubleType but was actually of type $actualDataType"
      )
    }

    // Checking group column type
    {
      val col = $(groupColPram)
      val actualDataType = schema(col).dataType
      require(
        actualDataType == StringType,
        s"Column $col must be of type StringType but was actually of type $actualDataType"
      )
    }

    // TODO: everything in $(fixedEffectColsParam) and $(randomEffectColsParam) must be doubles

    if ($(fixedEffectInterceptParam)) {
      // TODO: assert can not have the column "intercept"
    }

    if ($(randomEffectInterceptParam)) {
      // TODO: assert can not have the column "intercept"
    }

    // Appending prediction column if predicting
    if (!fitting) {
      val col = $(predictionColPram)
      require(!schema.fieldNames.contains(col), s"Column $col already exists.")
      StructType(schema.fields :+ StructField(col, DoubleType, nullable = false))
    } else {
      schema
    }
  }


  def transformDataset(dataset: Dataset[_]): Dataset[_] = {
    val fixedEffectsColName: String = "fixedEffects"
    val randomEffectsColName: String = "randomEffects"

    val datasetPlusFixed: Dataset[_] = addVectorFeature(dataset, $(fixedEffectColsParam).toArray, fixedEffectsColName,
      $(fixedEffectInterceptParam))
    val datasetPlusFixedRandom: Dataset[_] = addVectorFeature(datasetPlusFixed, $(randomEffectColsParam).toArray,
      randomEffectsColName, $(randomEffectInterceptParam))

    datasetPlusFixedRandom.select(col($(responseColPram)), col($(groupColPram)), col(fixedEffectsColName),
      col(randomEffectsColName))
  }


  def addVectorFeature(dataset: Dataset[_], colsArray: Array[String], featureName: String,
                       addIntercept: Boolean): Dataset[_] = {
    val interceptColumnName: String = "intercept"
    val finalColsArray: Array[String] = if (addIntercept) {
      Array(interceptColumnName) ++ colsArray
    } else {
      colsArray
    }
    val assembler: VectorAssembler = new VectorAssembler().
      setInputCols(finalColsArray).
      setOutputCol(featureName)

    val addInterceptUDF = udf { () => 1.0 }
    if (addIntercept) {
      assembler.transform(dataset.withColumn(interceptColumnName, addInterceptUDF()))
    } else {
      assembler.transform(dataset)
    }
  }

}
