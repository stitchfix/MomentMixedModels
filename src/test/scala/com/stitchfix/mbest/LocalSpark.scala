package com.stitchfix.mbest

import org.apache.spark.sql.SparkSession

trait LocalSpark {

  lazy val localSpark: SparkSession = {
    SparkSession
      .builder()
      .master("local")
      .appName("spark test example")
      .getOrCreate()
  }

}
