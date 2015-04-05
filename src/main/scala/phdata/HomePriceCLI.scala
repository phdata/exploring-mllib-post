package phdata

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.StandardScalerModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LinearRegressionModel
import io._

import scala.io.Source

object HomePriceCLI {
  def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setMaster("local[1]").setAppName("Home Price CLI"))
    val linRegModel = sc.objectFile[LinearRegressionModel]("linReg.model").first()
    val scalerModel = sc.objectFile[StandardScalerModel]("scaler.model").first()

    // home.age, home.bathrooms, home.bedrooms, home.garage, home.sqF
    println(linRegModel.predict(scalerModel.transform(Vectors.dense(11.0, 2.0, 2.0, 1.0, 2200.0))))
    sc.stop()
  }
}
