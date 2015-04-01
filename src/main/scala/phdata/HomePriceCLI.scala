package phdata

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LinearRegressionModel

object HomePriceCLI {
  def main(args: Array[String]): Unit = {

    val linReg = new LinearRegressionModel(intercept = 2, weights = Vectors.dense(1,2))

    println(linReg.predict(Vectors.dense(1,2)))
  }
}
