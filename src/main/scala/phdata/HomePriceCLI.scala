package phdata

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.StandardScalerModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LinearRegressionModel
import io._

import scala.io.Source

object HomePriceCLI {
  def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("Home Price CLI"))
    val linRegModel = sc.objectFile[LinearRegressionModel]("hdfs:///user/root/linReg.model").first()
    val scalerModel = sc.objectFile[StandardScalerModel]("hdfs:///user/root/scaler.model").first()
    /**
    val source = Source.fromFile("homeprice.model").mkString("").split(",").map(_.toDouble)


    val mean = source.take(5)
    val variance = source.drop(5).take(5)
    val intercept = source(11)
    val weights = source.drop(11)
    val linReg = new LinearRegressionModel(intercept = intercept, weights = Vectors.dense(weights.toArray))
    val scaler = new StandardScalerModel(Vectors.dense(variance), Vectors.dense(mean), true, true)
      **/

    // home.age, home.bathrooms, home.bedrooms, home.garage, home.sqF
    println(linRegModel.predict(scalerModel.transform(Vectors.dense(11.0, 2.0, 2.0, 1.0, 2200.0))))
  }
}
