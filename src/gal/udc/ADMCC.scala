package gal.udc

import scala.Ordering

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.numericRDDToDoubleRDDFunctions
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object sparkContextSingleton
{
  @transient private var instance: SparkContext = _
  private val conf : SparkConf = new SparkConf().setAppName("ADMCC")
                                                //.setMaster("local[8]")
                                                .setMaster("local[4]")
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


object ADMCC
{
  val DEFAULT_ANOMALY_RATIO:Double=0.03
  def showUsageAndExit()=
  {
    println("""Usage: ADMCC dataset [options]
    Dataset must be a libsvm file (labels are disregarded)
Options:
    -k    Intermediate parameter subspace dimension (default: """+ADMCCModel.DEFAULT_SUBSPACE_DIMENSION+""")
    -r    Regularization parameter (default: """+ADMCCModel.DEFAULT_REGULARIZATION_PARAMETER+""")
    -l0   Learning rate start (default: """+ADMCCModel.DEFAULT_LEARNING_RATE_START+""")
    -ls   Learning rate speed (default: """+ADMCCModel.DEFAULT_LEARNING_RATE_SPEED+""")
    -g    Number of gaussians in the GMM (default: """+ADMCCModel.DEFAULT_GAUSSIAN_COMPONENTS+""")
    -p    Anomaly ratio (default: """+DEFAULT_ANOMALY_RATIO+""")
    -nr    Normalizing radius (default: """+ADMCCModel.DEFAULT_NORMALIZING_R+""")
    -n    Maximum number of SGD iterations (default: """+ADMCCModel.DEFAULT_MAX_ITERATIONS+""")""")
    System.exit(-1)
  }
  def parseParams(p:Array[String]):Map[String, Any]=
  {
    val m=scala.collection.mutable.Map[String, Any]("subspace_dimension" -> ADMCCModel.DEFAULT_SUBSPACE_DIMENSION.toDouble,
                                                    "regularization_parameter" -> ADMCCModel.DEFAULT_REGULARIZATION_PARAMETER,
                                                    "learning_rate_start" -> ADMCCModel.DEFAULT_LEARNING_RATE_START,
                                                    "learning_rate_speed" -> ADMCCModel.DEFAULT_LEARNING_RATE_SPEED,
                                                    "first_continuous" -> ADMCCModel.DEFAULT_FIRST_CONTINUOUS.toDouble,
                                                    "minibatch" -> ADMCCModel.DEFAULT_MINIBATCH_SIZE.toDouble,
                                                    "gaussian_components" -> ADMCCModel.DEFAULT_GAUSSIAN_COMPONENTS.toDouble,
                                                    "max_iterations" -> ADMCCModel.DEFAULT_MAX_ITERATIONS.toDouble,
                                                    "anomaly_ratio" -> DEFAULT_ANOMALY_RATIO,
                                                    "normalizing_radius" -> ADMCCModel.DEFAULT_NORMALIZING_R)
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
          case "k"   => "subspace_dimension"
          case "l0"   => "learning_rate_start"
          case "ls"   => "learning_rate_speed"
          case "r"   => "regularization_parameter"
          case "fc"  => "first_continuous"
          case "g"  => "gaussian_components"
          case "n"  => "max_iterations"
          case "p"  => "anomaly_ratio"
          case "nr"  => "normalizing_radius"
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
    val admcc=new ADMCCModel()
    admcc.subspaceDimension=options("subspace_dimension").asInstanceOf[Double].toInt
    admcc.maxIterations=options("max_iterations").asInstanceOf[Double].toInt
    admcc.minibatchSize=options("minibatch").asInstanceOf[Double].toInt
    admcc.regParameter=options("regularization_parameter").asInstanceOf[Double]
    admcc.learningRate0=options("learning_rate_start").asInstanceOf[Double]
    admcc.learningRateSpeed=options("learning_rate_speed").asInstanceOf[Double]
    admcc.gaussianK=options("gaussian_components").asInstanceOf[Double].toInt
    admcc.normalizingR=options("normalizing_radius").asInstanceOf[Double]

    println("Training ADMC² model with parameters:\n\tG:"+admcc.gaussianK+" K:"+admcc.subspaceDimension+" R:"+admcc.normalizingR+" L0:"+admcc.learningRate0+" LS:"+admcc.learningRateSpeed+" NR:"+admcc.normalizingR+" N:"+admcc.maxIterations)
    
    //Training with the whole dataset
    admcc.trainWithSGD(sc, dataRDD.map(_._1), options("anomaly_ratio").asInstanceOf[Double])
    
    var anomaliesRDD=dataRDD.filter({x => admcc.isAnomaly(x._1)})
    
    println("The "+(Math.round(options("anomaly_ratio").asInstanceOf[Double]*10000)/100)+"% less probable elements are:")
    anomaliesRDD.foreach({x => println(x._1+","+x._2)})
  }
}