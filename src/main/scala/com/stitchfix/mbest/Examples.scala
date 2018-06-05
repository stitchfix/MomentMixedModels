package com.stitchfix.mbest

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import com.stitchfix.mbest.MixedEffectsRegression

import scala.collection.immutable.Seq

/**
  *
  */
object Examples {
  lazy val localSpark: SparkSession = {
    SparkSession
      .builder()
      .master("local")
      .appName("spark test example")
      .getOrCreate()
  }

  val sleepstudySchema = StructType(Array(
    StructField("Reaction", DoubleType, true),
    StructField("Days", DoubleType, true),
    StructField("Subject", StringType, true)))

  val sleepstudyData = {
    localSpark
      .read.format("csv")
      .option("header", "true")
      .schema(sleepstudySchema)
      .load("data/sleepstudy.csv")
  }


  val linearModelFitter = {
    new MixedEffectsRegression()
      .setResponseCol("Reaction")
      .setFixedEffectCols(Seq("Days"))
      .setRandomEffectCols(Seq("Days"))
      .setFamilyParam("gaussian")
      .setGroupCol("Subject")
  }

  val linearModel = linearModelFitter.fit(sleepstudyData)
  // fixed effects
  println(linearModel.β)
  // dispersion
  println(linearModel.φ)
  // random effects covariance
  println(linearModel.Σ)
  // random effects
  println(linearModel.randomEffects)


  val binomialSimSchema = StructType(Array(
    StructField("x1", DoubleType, true),
    StructField("x2", DoubleType, true),
    StructField("y", DoubleType, true),
    StructField("group_id", StringType, true)))

  val binomialSimData = {
    localSpark
      .read.format("csv")
      .option("header", "true")
      .schema(binomialSimSchema)
      .load("data/binomial_sim.csv")
  }

  val logisticModelFitter = {
    new MixedEffectsRegression()
      .setResponseCol("y")
      .setFixedEffectCols(Seq("x1", "x2"))
      .setRandomEffectCols(Seq("x1"))
      .setFamilyParam("binomial")
      .setGroupCol("group_id")
  }

  val logisticModel = logisticModelFitter.fit(binomialSimData)
  // fixed effects
  println(logisticModel.β)
  // dispersion
  println(logisticModel.φ)
  // random effects covariance
  println(logisticModel.Σ)
  // random effects
  println(logisticModel.randomEffects)

}
