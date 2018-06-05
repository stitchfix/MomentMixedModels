# MomentMixedModels

A Spark/Scala package for Moment-Based Estimation For Hierarchical Models. Based off the the
[mbest](https://cran.r-project.org/web/packages/mbest/index.html) R package, and
[(Patrick O. Perry 2015)](https://arxiv.org/abs/1504.04941). The package currently only supports linear and logistic
regression.

See the `Examples` object internally for examples of fitting local models.
```scala
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
  println(linearModel.β)
  // DenseVector(251.40510484848477, 10.467285959595985)
  println(linearModel.φ)
  // 654.9410270722987
  println(linearModel.Σ)
  // 565.5153668031181   11.055429022598702
  // 11.055429022598702  32.68219718640934
  println(linearModel.randomEffects("310"))
  // DenseVector(-38.43315312500708, -5.513378956616419)
```

## Future Development

The current documentation is quite sparse. We'll gladly accept requests to fix and improve documentation! We will also
gladly accept feature requestes and pull requests.

Please report issues directly on Github, that would be a really useful contribution given that we lack some user
testing for this project. Please document as much as possible the steps to reproduce your problem
(even better with screenshots).


## License

We’re using the [Apache 2.0](./LICENSE) license.

## Authors
- [Kyle Schmaus](https://github.com/kschmaus)
- [Alexander Demidko](https://github.com/xdralex)