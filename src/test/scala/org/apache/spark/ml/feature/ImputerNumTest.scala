package org.apache.spark.ml.feature

import org.scalatest.FunSuite
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}

class ImputerNumTest extends FunSuite {

  val conf: SparkConf = new SparkConf().setMaster("local[*]")
  val spark: SparkSession = SparkSession.builder.config(conf).getOrCreate()
  val sc: SparkContext = spark.sparkContext

  import spark.implicits._

  test("Test1. ImputerNum mean") {
    val input: DataFrame = Seq(
      (1.0, 2.0, 2.0, 1, "1a"),
      (2.0, 3.0, Double.NaN, 2, "2b"),
      (Double.NaN, 3.0, 2.0, 2, "2b")
    ).toDF("a", "b", "c", "target", "class")

    val expected: DataFrame = Seq(
      (1.0, 2.0, 2.0, 1, "1a", 1.0, 2.0, 2.0),
      (2.0, 3.0, Double.NaN, 2, "2b", 2.0, 3.0, 2.0),
      (Double.NaN, 3.0, 2.0, 2, "2b", 1.5, 3.0, 2.0)
    ).toDF("a", "b", "c", "target", "class", "a_imputed", "b_imputed", "c_imputed")

    val featureCols = List("a", "b", "c").toArray
    val imputer = new ImputerNum()
      .setStrategy("mean")
      .setOutputCols(featureCols.map(_ + "_imputed"))
      .setInputCols(featureCols)

    val output = imputer.fit(input).transform(input)

    assert(output.toJSON.rdd.collect().mkString(";") == expected.toJSON.rdd.collect().mkString(";"))
  }

  test("Test2. ImputerNum zero") {
    val input: DataFrame = Seq(
      (1.0, 2.0, 2.0, 1, "1a"),
      (2.0, 3.0, Double.NaN, 2, "2b"),
      (Double.NaN, 3.0, 2.0, 2, "2b")
    ).toDF("a", "b", "c", "target", "class")

    val expected: DataFrame = Seq(
      (1.0, 2.0, 2.0, 1, "1a", 1.0, 2.0, 2.0),
      (2.0, 3.0, Double.NaN, 2, "2b", 2.0, 3.0, 0.0),
      (Double.NaN, 3.0, 2.0, 2, "2b", 0.0, 3.0, 2.0)
    ).toDF("a", "b", "c", "target", "class", "a_imputed", "b_imputed", "c_imputed")

    val featureCols = List("a", "b", "c").toArray
    val imputer = new ImputerNum()
      .setStrategy("const")
      .setOutputCols(featureCols.map(_ + "_imputed"))
      .setInputCols(featureCols)

    val output = imputer.fit(input).transform(input)
    output.show()

    assert(output.toJSON.rdd.collect().mkString(";") == expected.toJSON.rdd.collect().mkString(";"))
  }

  test("Test3. ImputerNum const") {
    val input: DataFrame = Seq(
      (1.0, 2.0, 2.0, 1, "1a"),
      (2.0, 3.0, Double.NaN, 2, "2b"),
      (Double.NaN, 3.0, 2.0, 2, "2b")
    ).toDF("a", "b", "c", "target", "class")

    val expected: DataFrame = Seq(
      (1.0, 2.0, 2.0, 1, "1a", 1.0, 2.0, 2.0),
      (2.0, 3.0, Double.NaN, 2, "2b", 2.0, 3.0, -999.0),
      (Double.NaN, 3.0, 2.0, 2, "2b", -999.0, 3.0, 2.0)
    ).toDF("a", "b", "c", "target", "class", "a_imputed", "b_imputed", "c_imputed")

    val featureCols = List("a", "b", "c").toArray
    val imputer = new ImputerNum()
      .setStrategy("const")
      .setOutputCols(featureCols.map(_ + "_imputed"))
      .setInputCols(featureCols)
      .setFillValue(-999.0)

    val output = imputer.fit(input).transform(input)
    output.show()

    assert(output.toJSON.rdd.collect().mkString(";") == expected.toJSON.rdd.collect().mkString(";"))
  }
}