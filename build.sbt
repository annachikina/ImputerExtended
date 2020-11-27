name := "Imputer"

version := "1.0.0"

scalaVersion := "2.11.12"

val dependencies = new {
  private val sparkVersion = "2.4.3"
  val sparkMlLib = "org.apache.spark" %% "spark-mllib" % sparkVersion % Compile
  val scalaTest = "org.scalatest" %% "scalatest" % "3.0.8" % Test
}

libraryDependencies ++= Seq(
  dependencies.sparkMlLib,
  dependencies.scalaTest
)

parallelExecution in Test := false

