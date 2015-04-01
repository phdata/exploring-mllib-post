package phdata

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.{SparkContext, SparkConf}

case class Home(mlsNum: Double, city: String, sqFt: Double, bedrooms: Double, bathrooms: Double,
                    garage: Double, age: Double, acres: Double, price: Double)

object HomePriceRecommender extends Serializable {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("Home Price Recommender"))
    val base = "hdfs:///user/root/homeprice.data"
    val homeData = sc.textFile(base)


    val parsed = homeData.map(line => parse(line))

    // look at some statistics of the data
    val priceStats = Statistics.colStats(parsed.map(home => Vectors.dense(home.price)))
    println("Price mean: " + priceStats.mean)
    println("Price max: " + priceStats.max)
    println("Price min: " + priceStats.max)

    // filter out anomalous data
    val filtered = parsed.filter(home => (home.price > 100000.0 && home.price < 400000.0 && home.sqFt > 1000.0))

    // see how correlated price and square feet are
    val corr = Statistics.corr(filtered.map(home => home.price), filtered.map(home => home.sqFt))
    println("Price and square feet corr: " + corr)

    // convert to labeled data for MLLib
    val labelData = filtered.map { home =>
      LabeledPoint(home.price, Vectors.dense(home.age, home.bathrooms,
        home.bedrooms, home.garage, home.sqFt))
    }.cache()

    // Scale features to 0 mean and common variance
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(labelData.map(x => x.features))

    val scaledData = labelData.map{ data =>
      LabeledPoint(data.label, scaler.transform(Vectors.dense(data.features.toArray)))
    }

    val numIterations = 1000
    val stepSize = 0.2
    // Setup linear regression model and ensure it finds the intercept
    val linearReg = new LinearRegressionWithSGD()
    linearReg.setIntercept(true)
    linearReg.optimizer
      .setNumIterations(numIterations)
      .setStepSize(stepSize)

    // run linear regresssion
    val model = linearReg.run(scaledData)

    println("Model: " + model)

    // determine how well the model predicts the trained data's home prices
    val valuesAndPreds = scaledData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val power = valuesAndPreds.map{
      case(v, p) => math.pow((v - p), 2)
    }

    // Mean Square Error
    val MSE = power.reduce((a, b) => a + b) / power.count()

    println("Mean Square Error: " + MSE)

    // persist model to HDFS
    val conf = new Configuration()
    val fs = FileSystem.get(conf)
    val out = fs.create( new Path("hdfs:///user/root/homeprice.model"))
    out.writeBytes(model.intercept.toString)
    out.writeBytes(",")
    out.writeBytes(model.weights.toArray.mkString(","))
    out.close()
    fs.close()
  }

  // parse home price data into case class
  def parse(line: String) = {
    val split = line.split('|')
    val mlsNum = split(0).toDouble
    val city = split(1).toString
    val sqFt = split(2).toDouble
    val bedrooms = split(3).toDouble
    val bathrooms = split(4).toDouble
    val garage = split(5).toDouble
    val age = split(6).toDouble
    val acres = split(7).toDouble
    val price = split(8).toDouble
    Home(mlsNum, city, sqFt, bedrooms, bathrooms, garage, age, acres, price)
  }
}
