package gal.udc

import scala.Ordering

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.numericRDDToDoubleRDDFunctions
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import breeze.linalg.DenseVector
import org.apache.log4j.Level
import org.apache.log4j.Logger

object sparkContextSingleton
{
  /*DEBUG*/
    //Stop annoying INFO messages
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    
  val NUMBER_OF_CORES=1
  @transient private var instance: SparkContext = _
  private val conf : SparkConf = new SparkConf()//.setAppName("ADMNC")
                                                //.setMaster("local["+NUMBER_OF_CORES+"]")
                                                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                                                .set("spark.broadcast.factory", "org.apache.spark.broadcast.HttpBroadcastFactory")
                                                //.set("spark.eventLog.enabled", "true")
                                                //.set("spark.eventLog.dir","file:///home/eirasf/Escritorio/datasets-anomalias/sparklog-local")
                                                .set("spark.kryoserializer.buffer.max", "512")
                                                .set("spark.driver.maxResultSize", "2048")

  def getInstance(): SparkContext=
  {
    if (instance == null)
      instance = new SparkContext(conf)
    instance
  }
}


object ADMNC
{
  val DEFAULT_ANOMALY_RATIO:Double=0.03
  def showUsageAndExit()=
  {
    println("""Usage: ADMNC dataset [options]
    Dataset must be a libsvm file (labels are disregarded)
Options:
    -fc   Index of the first numeric variable (mandatory)
    -p    Anomaly ratio (default: """+DEFAULT_ANOMALY_RATIO+""")

  Hyperparameters
    -g    Number of gaussians in the GMM (default: """+ADMNC_LogisticModel.DEFAULT_GAUSSIAN_COMPONENTS+""")

  Advanced hyperparameters
    -r    Regularization parameter (default: """+ADMNC_LogisticModel.DEFAULT_REGULARIZATION_PARAMETER+""")
    -n    Maximum number of SGD iterations (default: """+ADMNC_LogisticModel.DEFAULT_MAX_ITERATIONS+""")""")
    System.exit(-1)
  }
  def parseParams(p:Array[String]):Map[String, Any]=
  {
    val m=scala.collection.mutable.Map[String, Any]("regularization_parameter" -> ADMNC_LogisticModel.DEFAULT_REGULARIZATION_PARAMETER,
                                                    "first_continuous" -> ADMNC_LogisticModel.DEFAULT_FIRST_CONTINUOUS.toDouble,
                                                    "minibatch" -> ADMNC_LogisticModel.DEFAULT_MINIBATCH_SIZE.toDouble,
                                                    "gaussian_components" -> ADMNC_LogisticModel.DEFAULT_GAUSSIAN_COMPONENTS.toDouble,
                                                    "max_iterations" -> ADMNC_LogisticModel.DEFAULT_MAX_ITERATIONS.toDouble,
                                                    "anomaly_ratio" -> DEFAULT_ANOMALY_RATIO)
    if (p.length<=0)
      showUsageAndExit()
    
    m("dataset")=p(0)
    
    var i=1
    while (i < p.length)
    {
      if ((i>=p.length-1) || (p(i).charAt(0)!='-'))
      {
        println("Unknown option: "+p(i))
        showUsageAndExit()
      }
      val readOptionName=p(i).substring(1)
      val option=readOptionName match
        {
          case "r"   => "regularization_parameter"
          case "fc"  => "first_continuous"
          case "g"  => "gaussian_components"
          case "n"  => "max_iterations"
          case "p"  => "anomaly_ratio"
          case somethingElse => readOptionName
        }
      if (!m.keySet.exists(_==option))
      {
        println("Unknown option:"+readOptionName)
        showUsageAndExit()
      }
      m(option)=p(i+1).toDouble
      i=i+2
    }
    return m.toMap
  }
  def main(args: Array[String])
  {
    val options=parseParams(args)
    val sc=sparkContextSingleton.getInstance()
    
    //Load and transform dataset
    println("Loading dataset "+options("dataset").asInstanceOf[String])
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, options("dataset").asInstanceOf[String])
    val firstContinuous=options("first_continuous").asInstanceOf[Double].toInt - 1
    
    val dataRDD=MixedData.fromLabeledPointRDD(data, firstContinuous, true)
    dataRDD.cache()
    
    //Model configuration, creation and training
    val admnc=new ADMNC_LogisticModel()
    admnc.maxIterations=options("max_iterations").asInstanceOf[Double].toInt
    admnc.minibatchSize=options("minibatch").asInstanceOf[Double].toInt
    admnc.regParameter=options("regularization_parameter").asInstanceOf[Double]
    admnc.gaussianK=options("gaussian_components").asInstanceOf[Double].toInt

    println("Training ADMNC model with parameters:\n\tG:"+admnc.gaussianK+" N:"+admnc.maxIterations)
    
    //Training with the whole dataset
    admnc.trainWithSGD(sc, dataRDD.map(_._1), options("anomaly_ratio").asInstanceOf[Double])
    
    var anomaliesRDD=dataRDD.filter({x => admnc.isAnomaly(x._1)})
    
    println("The "+(Math.round(options("anomaly_ratio").asInstanceOf[Double]*10000)/100)+"% less probable elements are:")
    anomaliesRDD.foreach({x => println(x._1+","+x._2)})
  }
}