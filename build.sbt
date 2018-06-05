organization := "com.stitchfix"

organizationName := "Stitch Fix"
// author := ("Kyle Schmaus", "Alexander Demidko")
startYear := Some(2011)
licenses += ("Apache-2.0", new URL("https://www.apache.org/licenses/LICENSE-2.0.txt"))

name := "mbest"

scalaVersion := "2.11.11"

lazy val root = project.in(file("."))

resolvers += "StitchFix releases" at "http://artifactory.vertigo.stitchfix.com/artifactory/releases"
resolvers += "Artima Maven Repository" at "http://repo.artima.com/releases"

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.2",

  "org.apache.spark" %% "spark-core" % "2.2.0",
  "org.apache.spark" %% "spark-mllib" % "2.2.0",
  "org.apache.spark" %% "spark-sql" % "2.2.0",
  "org.apache.spark" %% "spark-hive" % "2.2.0",
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "com.stitchfix.algorithms.spark" %% "sfs3" % "0.7.0-spark2.2.0",

  "org.scalatest" %% "scalatest" % "3.0.4",
  "org.scalatest" %% "scalatest" % "3.0.4" % "test"
)
