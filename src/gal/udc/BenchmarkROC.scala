package gal.udc

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions

import breeze.linalg.{ DenseVector => BDV }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import java.io.PrintWriter
import java.io.File
import org.apache.spark.mllib.linalg.Vectors


object BenchmarkROC
{
  val DEFAULT_SUBSPACE_DIMENSION:Int=2
  val DEFAULT_REGULARIZATION_PARAMETER:Double=0.01
  val DEFAULT_LEARNING_RATE_START:Double=1.0
  val DEFAULT_LEARNING_RATE_SPEED:Double=0.1
  val DEFAULT_FIRST_CONTINUOUS:Int=2
  val DEFAULT_MINIBATCH_SIZE:Int=100
  val DEFAULT_MAX_ITERATIONS:Int=50
  val DEFAULT_GAUSSIAN_COMPONENTS:Int=2
  val DEFAULT_OUTPUT_FILE:String="out.txt"
  
  def showUsageAndExit()=
  {
    println("""Usage: ADMCC dataset [options]
    Dataset must be a libsvm file
Options:
    -k    Intermediate parameter subspace dimension (default: """+DEFAULT_SUBSPACE_DIMENSION+""")
    -r    Regularization parameter (default: """+DEFAULT_REGULARIZATION_PARAMETER+""")
    -l0   Learning rate start (default: """+DEFAULT_LEARNING_RATE_START+""")
    -ls   Learning rate speed (default: """+DEFAULT_LEARNING_RATE_SPEED+""")
    -g    Number of gaussians in the GMM (default: """+DEFAULT_GAUSSIAN_COMPONENTS+""")
    -n    Maximum number of SGD iterations (default: """+DEFAULT_MAX_ITERATIONS+""")
    -o    Output file (default: """+DEFAULT_OUTPUT_FILE+""")""")
    System.exit(-1)
  }
  def parseParams(p:Array[String]):Map[String, Any]=
  {
    val m=scala.collection.mutable.Map[String, Any]("subspace_dimension" -> DEFAULT_SUBSPACE_DIMENSION.toDouble,
                                                    "regularization_parameter" -> DEFAULT_REGULARIZATION_PARAMETER,
                                                    "learning_rate_start" -> DEFAULT_LEARNING_RATE_START,
                                                    "learning_rate_speed" -> DEFAULT_LEARNING_RATE_SPEED,
                                                    "first_continuous" -> DEFAULT_FIRST_CONTINUOUS.toDouble,
                                                    "minibatch" -> DEFAULT_MINIBATCH_SIZE.toDouble,
                                                    "gaussian_components" -> DEFAULT_GAUSSIAN_COMPONENTS.toDouble,
                                                    "max_iterations" -> DEFAULT_MAX_ITERATIONS.toDouble,
                                                    "output_file" -> DEFAULT_OUTPUT_FILE)
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
          case "o"  => "output_file"
          case somethingElse => readOptionName
        }
      if (!m.keySet.exists(_==option))
      {
        println("Unknown option:"+readOptionName)
        showUsageAndExit()
      }
      if (option=="output_file")
        m(option)=p(i+1)
      else
        m(option)=p(i+1).toDouble
      i=i+2
    }
    return m.toMap
  }
  def getMixedDataRDDFromLabeledPointRDD(lpRDD:RDD[LabeledPoint], firstContinuous:Int):RDD[(MixedData,Boolean)]=
  {
    val maxs:Array[Int]=lpRDD.map({ x => x.features.toArray.slice(0, firstContinuous) })
                                .reduce({ (x,y) =>x.zip(y).map({case (a,b) => Math.max(a,b)})})
                                .map({ x => if (x>1) x.toInt + 1 else 1})
    return lpRDD.map({ x => (new MixedData(BDV(x.features.toArray).slice(firstContinuous, x.features.size), BDV(x.features.toArray.slice(0, firstContinuous)), maxs), x.label==1) })
  }
  def main(args: Array[String])
  {
    val options=parseParams(args)
    val sc=sparkContextSingleton.getInstance()
    
    //Load and transform dataset
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, options("dataset").asInstanceOf[String])
    
    /* DATASET FIX
     * 
     * val reducedFeatures:Array[Double]=Array(1-x.features(0), x.features(2), x.features(3)) //Matlab-artificial
     * val reducedFeatures:Array[Double]=Array(x.features(1)+x.features(2)*2, x.features(3), x.features(4), x.features(5), x.features(6), x.features(7), x.features(8), x.features(9)) //Arritmia
     * val reducedFeatures:Array[Double]=Array(x.features(0)-1,x.features(1)-1, x.features(2)-1, x.features(3)-1, x.features(4)-1, x.features(5)-1, x.features(6)-1, x.features(7)-1, x.features(8)-1, x.features(9)-1, x.features(10)-1, x.features(11)-1, x.features(12)-1, x.features(13), x.features(14), x.features(15), x.features(16), x.features(17), x.features(18), x.features(19)) //german_statlog
     * 
    MLUtils.saveAsLibSVMFile(data.map{{ x => val reducedFeatures:Array[Double]=Array(x.features(0)-1,x.features(1)-1, x.features(2)-1, x.features(3)-1, x.features(4)-1, x.features(5)-1, x.features(6)-1, x.features(7)-1, x.features(8)-1, x.features(9)-1, x.features(10)-1, x.features(11)-1, x.features(12)-1, x.features(13), x.features(14), x.features(15), x.features(16), x.features(17), x.features(18), x.features(19))
                                             new LabeledPoint(x.label, Vectors.dense(reducedFeatures)) }}, "/home/eirasf/Escritorio/german_statlog-nodisc.libsvm")
    System.exit(0);
    * 
    */
    
    val firstContinuous=options("first_continuous").asInstanceOf[Double].toInt - 1
    val dataRDD=MixedData.fromLabeledPointRDD(data, firstContinuous, true)
    //val dataRDD=MixedData.fromLabeledPointRDD(data, firstContinuous, false)
    
    val sampleElement=dataRDD.first
    println("Sample element:"+sampleElement)
    
    dataRDD.cache()
    
    //val folds=MLUtils.kFold(dataRDD, 10, System.nanoTime().toInt)
    var folds=MLUtils.kFold(dataRDD, 2, System.nanoTime().toInt)
    
    //val regPars=Array[Double](0.001, 0.01, 0.1, 1, 10.0, 100)
    val regPars=Array[Double](1)
    //val estimatedPercentages=Array[Double](0.01, 0.03)
    //val subspaceDimensions=Array[Int](2, 4, 6)
    val subspaceDimensions=Array[Int](4)
    //val numGaussians=Array[Int](2, 4, 6)
    val numGaussians=Array[Int](4)
    //val lambdas=Array[Double](0.01, 0.1, 1, 10, 100)
    val lambdas=Array[Double](0.01)
    //val lambdaSpeeds=Array[Double](0.001, 0.01, 0.1, 1, 10, 100)
    val lambdaSpeeds=Array[Double](0.001)
    
    //var (bestResult, bestR, bestP, bestSD, bestNG) = (1.0, regPars(0), estimatedPercentages(0), subspaceDimensions(0), numGaussians(0))
    var (bestResult, bestR, bestL0, bestLS, bestG, bestSD) = (0.0, regPars(0), lambdas(0), lambdaSpeeds(0), numGaussians(0), subspaceDimensions(0))
    var (worstResult, worstR, worstL0, worstLS, worstG, worstSD) = (2.0, regPars(0), lambdas(0), lambdaSpeeds(0), numGaussians(0), subspaceDimensions(0))
    
    val pw = new PrintWriter(new File(options("output_file").asInstanceOf[String]))
    
    //TODO - Do this the Spark 2.0 way: with Dataset, ParamGrid and such,
    for (g <- numGaussians)
      for (sD <- subspaceDimensions)
        for (regP <- regPars)
          for (l0 <- lambdas)
            for (ls <- lambdaSpeeds)
            {
              pw.println("R:"+regP+" L0:"+l0+" LS:"+ls+" G:"+g+" K:"+sD)
              println("R:"+regP+" L0:"+l0+" LS:"+ls+" G:"+g+" K:"+sD)
              var modelAUROC=0.0
              var fCount=0
              //folds=MLUtils.kFold(dataRDD, 2, System.nanoTime().toInt)
              for (fold <- folds)
              {
                fCount=fCount+1
                pw.println("Fold:"+fCount)
                //Model configuration, creation and training
                val admcc=new ADMNCModel()
                admcc.subspaceDimension=sD
                admcc.maxIterations=options("max_iterations").asInstanceOf[Double].toInt
                admcc.minibatchSize=options("minibatch").asInstanceOf[Double].toInt
                admcc.regParameter=regP
                admcc.learningRate0=l0
                admcc.learningRateSpeed=ls
                admcc.gaussianK=g
                
                admcc.minibatchSize=90
                
                val training=fold._1//.sample(false, 0.55, System.nanoTime().toInt)
                
                //admcc.trainWithSGD(sc, fold._1.map(_._1), perc)
                admcc.trainWithSGD(sc, training.filter(!_._2).map(_._1), 0.01)

                
                //val test=fold._2.sample(false, 0.2, System.nanoTime().toInt)
                val test=fold._2
                println("TRAIN:"+training.count()+" TEST:"+test.count())
                
                val probs=test.map({case (element, label) =>(admcc.getProbabilityEstimator(element), if (label) 0.0 else 1.0)
                                                                           })//Mark as possitive non-anomalous elements, which are given high estimators.
                //val minMax=probs.map({case (es, cl) => (es,es)}).reduce({case ((max1,min1),(max2,min2)) => (Math.max(max1, max2),Math.min(min1, min2))})        
                //val metrics = new BinaryClassificationMetrics(probs.map({case (es,cl) => ((es-minMax._2)/(minMax._1-minMax._2),cl)}))
                val metrics = new BinaryClassificationMetrics(probs)
                // ROC Curve
                val roc = metrics.roc
                
                // AUROC
                val auROC = metrics.areaUnderROC
                
                //var probabilityRDD=admcc.getProbabilityEstimator(dataRDD.map(_._1))
                /*var misclassificationRate=fold._2.map({case (element, label) => val result=admcc.isAnomaly(element)
                                                               if (result==label)
                                                                 0.0
                                                               else
                                                                 1.0
                                                               }).sum()/fold._2.count()*/
                modelAUROC=modelAUROC+auROC
                pw.println("Fold AUROC:"+auROC);
              }
              modelAUROC=modelAUROC/folds.length.toDouble
              pw.println("AUROC:"+modelAUROC)
              pw.flush()
              println("AUROC:"+modelAUROC)
              if (modelAUROC>bestResult)
              {
                bestResult=modelAUROC
                bestR=regP
                bestL0=l0
                bestLS=ls
                bestG=g
                bestSD=sD
              }
              if (modelAUROC<worstResult)
              {
                worstResult=modelAUROC
                worstR=regP
                worstL0=l0
                worstLS=ls
                worstG=g
                worstSD=sD
              }
          }
    pw.println("BEST - R:"+bestR+" L0:"+bestL0+" LS:"+bestLS+" G:"+bestG+" K:"+bestSD+" obtained "+bestResult+" MCR")
    pw.println("WORST - R:"+worstR+" L0:"+worstL0+" LS:"+worstLS+" G:"+worstG+" K:"+worstSD+" obtained "+worstResult+" MCR")
    pw.close
  }
}