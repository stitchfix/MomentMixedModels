package com.stitchfix.mbest

import com.stitchfix.mbest.MixedEffectsRegression._
import com.twitter.chill.Kryo
import org.apache.spark.SparkConf
import org.apache.spark.serializer.{KryoRegistrator, KryoSerializer}

class MBestRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo): Unit = {
    kryo.register(classOf[Instance])
    kryo.register(classOf[Data])
    kryo.register(classOf[Estimate_Σ])
    kryo.register(classOf[Estimate_η0])
    kryo.register(classOf[Estimate_φ])
    kryo.register(classOf[Estimate_β])
  }
}

object MBestRegistrator {
  def register(conf: SparkConf) {
    conf.set("spark.serializer", classOf[KryoSerializer].getName)
    conf.set("spark.kryo.registrator", classOf[MBestRegistrator].getName)
  }
}