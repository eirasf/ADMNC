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
import java.util.Date
import java.sql.Timestamp
import org.apache.spark.mllib.linalg.DenseVector


object BenchmarkROC
{
  val DEFAULT_SUBSPACE_DIMENSION:Array[Int]=Array[Int](ADMNC_LogisticModel.DEFAULT_SUBSPACE_DIMENSION)
  val DEFAULT_REGULARIZATION_PARAMETER:Array[Double]=Array[Double](0.1, 1, 10.0, 100.0, 1000.0)
  val DEFAULT_LEARNING_RATE_START:Array[Double]=Array[Double](ADMNC_LogisticModel.DEFAULT_LEARNING_RATE_START)
  val DEFAULT_LEARNING_RATE_SPEED:Array[Double]=Array[Double](0.0001, 0.001, 0.01, 0.1, 1.0)
  val DEFAULT_FIRST_CONTINUOUS:Int=2
  val DEFAULT_NUMBER_OF_FOLDS:Int=10
  val DEFAULT_MINIBATCH_SIZE:Int=ADMNC_LogisticModel.DEFAULT_MINIBATCH_SIZE
  val DEFAULT_MAX_ITERATIONS:Int=ADMNC_LogisticModel.DEFAULT_MAX_ITERATIONS
  val DEFAULT_GAUSSIAN_COMPONENTS:Array[Int]=Array[Int](ADMNC_LogisticModel.DEFAULT_GAUSSIAN_COMPONENTS)
  val DEFAULT_LOGISTIC_LAMBDA:Array[Double]=Array[Double](ADMNC_LogisticModel.DEFAULT_LOGISTIC_LAMBDA)
  val DEFAULT_NORMALIZING_R:Array[Double]=Array[Double](ADMNC_LogisticModel.DEFAULT_NORMALIZING_R)
  val DEFAULT_OUTPUT_FILE:String="out.txt"
  val DEFAULT_TRAINING_SIZE_FOR_MEASURE:Int=500
  
  def showUsageAndExit()=
  {
    println("""Usage: BenchmarkROC dataset [options]
    Dataset must be a libsvm file
Options:
    -k    Intermediate parameter subspace dimension (default: """+DEFAULT_SUBSPACE_DIMENSION+""")
    -r    Regularization parameter (default: """+DEFAULT_REGULARIZATION_PARAMETER+""")
    -l0   Learning rate start (default: """+DEFAULT_LEARNING_RATE_START+""")
    -ls   Learning rate speed (default: """+DEFAULT_LEARNING_RATE_SPEED+""")
    -fc   Index of the first continuous variable (default: """+DEFAULT_FIRST_CONTINUOUS+""")
    -g    Number of gaussians in the GMM (default: """+DEFAULT_GAUSSIAN_COMPONENTS+""")
    -n    Maximum number of SGD iterations (default: """+DEFAULT_MAX_ITERATIONS+""")
    -ll   Normalizing factor of the logistic function (default: """+DEFAULT_LOGISTIC_LAMBDA+""")
    -nr   Normalizing radius for the subspace projections (default: """+DEFAULT_NORMALIZING_R+""")
    -o    Output file (default: """+DEFAULT_OUTPUT_FILE+""")""")
    System.exit(-1)
  }
  def parseParams(p:Array[String]):Map[String, Any]=
  {
    val m=scala.collection.mutable.Map[String, Any]("subspace_dimension" -> DEFAULT_SUBSPACE_DIMENSION,
                                                    "regularization_parameter" -> DEFAULT_REGULARIZATION_PARAMETER,
                                                    "learning_rate_start" -> DEFAULT_LEARNING_RATE_START,
                                                    "learning_rate_speed" -> DEFAULT_LEARNING_RATE_SPEED,
                                                    "first_continuous" -> DEFAULT_FIRST_CONTINUOUS.toDouble,
                                                    "minibatch" -> DEFAULT_MINIBATCH_SIZE.toDouble,
                                                    "gaussian_components" -> DEFAULT_GAUSSIAN_COMPONENTS,
                                                    "max_iterations" -> DEFAULT_MAX_ITERATIONS.toDouble,
                                                    "folds" -> DEFAULT_NUMBER_OF_FOLDS.toDouble,
                                                    "logistic_lambda" -> DEFAULT_LOGISTIC_LAMBDA,
                                                    "normalizing_radius" -> DEFAULT_NORMALIZING_R,
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
          case "k"  => "subspace_dimension"
          case "l0" => "learning_rate_start"
          case "ls" => "learning_rate_speed"
          case "r"  => "regularization_parameter"
          case "fc" => "first_continuous"
          case "g"  => "gaussian_components"
          case "n"  => "max_iterations"
          case "f"  => "folds"
          case "o"  => "output_file"
          case "ll" => "logistic_lambda"
          case "nr" => "normalizing_radius"
          case somethingElse => readOptionName
        }
      if (!m.keySet.exists(_==option))
      {
        println("Unknown option:"+readOptionName)
        showUsageAndExit()
      }
      
      option match
      {
        case "output_file" => m(option)=p(i+1)
        case "regularization_parameter" |
             "learning_rate_start" |
             "learning_rate_speed" |
             "logistic_lambda" |
             "normalizing_radius" => m(option)=p(i+1).split(",").map { x => x.toDouble }
        case "subspace_dimension" |
             "gaussian_components" => m(option)=p(i+1).split(",").map { x => x.toInt }
        case somethingElse => m(option)=p(i+1).toDouble
      }
      
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
    
    val filename=options("dataset").asInstanceOf[String]
    
    //Load and transform dataset
    val trainingData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, filename)
    val devData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, filename.replace("train", "dev"))
    val testData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, filename.replace("train", "test"))
    
    val firstContinuous=options("first_continuous").asInstanceOf[Double].toInt - 1
    val trainingDataRDD=MixedData.fromLabeledPointRDD(trainingData, firstContinuous, true)
    val devDataRDD=MixedData.fromLabeledPointRDD(devData, firstContinuous, true)
    val testDataRDD=MixedData.fromLabeledPointRDD(testData, firstContinuous, true)
    //val dataRDD=MixedData.fromLabeledPointRDD(data, firstContinuous, false)
    
    val sampleElement=trainingDataRDD.first
    println("Sample element:"+sampleElement)
    
    trainingDataRDD.cache()

    val folds=MLUtils.kFold(trainingDataRDD, options("folds").asInstanceOf[Double].toInt, System.nanoTime().toInt)
    
    val regPars=options("regularization_parameter").asInstanceOf[Array[Double]]
    val subspaceDimensions=options("subspace_dimension").asInstanceOf[Array[Int]]
    val numGaussians=options("gaussian_components").asInstanceOf[Array[Int]]
    val lambdas=options("learning_rate_start").asInstanceOf[Array[Double]]
    val lambdaSpeeds=options("learning_rate_speed").asInstanceOf[Array[Double]]
    val logisticLambdas=options("logistic_lambda").asInstanceOf[Array[Double]]
    val normalizingRadiuses=options("normalizing_radius").asInstanceOf[Array[Double]]
    
    //var (bestResult, bestR, bestP, bestSD, bestNG) = (1.0, regPars(0), estimatedPercentages(0), subspaceDimensions(0), numGaussians(0))
    var (bestResult, bestR, bestL0, bestLS, bestG, bestSD, bestLL, bestNR) = (0.0, regPars(0), lambdas(0), lambdaSpeeds(0), numGaussians(0), subspaceDimensions(0), logisticLambdas(0), normalizingRadiuses(0))
    var (worstResult, worstR, worstL0, worstLS, worstG, worstSD, worstLL, worstNR) = (2.0, regPars(0), lambdas(0), lambdaSpeeds(0), numGaussians(0), subspaceDimensions(0), logisticLambdas(0), normalizingRadiuses(0))
    
    val pw = new PrintWriter(new File(options("output_file").asInstanceOf[String]))
    
    pw.println("Hyperparameter optimization grid over "+filename)
    pw.println("Parameters explored:")
    pw.println("\tRegularization parameter: "+regPars.mkString(" | "))
    pw.println("\tSubspace dimension: "+subspaceDimensions.mkString(" | "))
    pw.println("\tNumber of gaussians: "+numGaussians.mkString(" | "))
    pw.println("\tLambda_0: "+lambdas.mkString(" | "))
    pw.println("\tLambda speed: "+lambdaSpeeds.mkString(" | "))
    pw.println("\tLogistic lambda: "+logisticLambdas.mkString(" | "))
    pw.println("\tNormalizing radius: "+normalizingRadiuses.mkString(" | "))
    pw.println("Number of CV folds: "+folds.length)

    var testCount=0
    val totalTests=regPars.length*subspaceDimensions.length*numGaussians.length*lambdas.length*lambdaSpeeds.length*logisticLambdas.length*normalizingRadiuses.length
    
    var i=0
    pw.println("Fold descriptions:\nFold number\t#TRAINING\t#TEST")
    println("Fold descriptions:\nFold number\t#TRAINING\t#TEST")
    for (fold <- folds)
    {
      pw.println(i+"\t"+fold._1.count()+"\t"+fold._2.count())
      println(i+"\t"+fold._1.count()+"\t"+fold._2.count())
      i+=1
    }

    val startTime=System.currentTimeMillis()
    pw.println("Starting at "+new Timestamp(new Date().getTime())+" using "+sparkContextSingleton.NUMBER_OF_CORES+" cores")
    
    //TODO - Do this the Spark 2.0 way: with Dataset, ParamGrid and such,
    for (g <- numGaussians)
      for (sD <- subspaceDimensions)
        for (regP <- regPars)
          for (l0 <- lambdas)
            for (ls <- lambdaSpeeds)
              for (ll <- logisticLambdas)
                for (nr <- normalizingRadiuses)
                {
                  var modelAUROC=0.0
                  var modelTrainingAUROC=0.0
                  var fCount=0
                  testCount+=1
                  pw.println("["+testCount+"/"+totalTests+"] - R:"+regP+" L0:"+l0+" LS:"+ls+" G:"+g+" K:"+sD+" LL:"+ll+" NR:"+nr)
                  println("["+testCount+"/"+totalTests+"] - R:"+regP+" L0:"+l0+" LS:"+ls+" G:"+g+" K:"+sD+" LL:"+ll+" NR:"+nr)
                for (fold <- folds)
                { 
                  //pw.println("Fold:"+fCount)
                  //Model configuration, creation and training
                  val admnc=new ADMNC_LogisticModel()
                  admnc.maxIterations=options("max_iterations").asInstanceOf[Double].toInt
                  admnc.minibatchSize=options("minibatch").asInstanceOf[Double].toInt
                  admnc.regParameter=regP
                  admnc.learningRate0=l0
                  admnc.learningRateSpeed=ls
                  admnc.gaussianK=g
                  admnc.logisticLambda=ll
                  
                  //admnc.minibatchSize=90
                  //admnc.maxIterations=1000
                  
                  val training=fold._1//.sample(false, 0.55, System.nanoTime().toInt)
                  
                  //admnc.trainWithSGD(sc, fold._1.map(_._1), perc)
                  admnc.trainWithSGD(sc, training.filter(!_._2).map(_._1), 0.01)

                  
                  //val test=fold._2.sample(false, 0.2, System.nanoTime().toInt)
                  val test=fold._2
                  //println("TRAIN:"+training.count()+" TEST:"+test.count())
                  
                  //TEST AUROC
                  val probs=test.map({case (element, label) =>(admnc.getProbabilityEstimator(element), if (label) 0.0 else 1.0)
                                                                            })//Mark as possitive non-anomalous elements, which are given high estimators.
                  var foldAuroc = new BinaryClassificationMetrics(probs).areaUnderROC
                  
                  var convergenceInfo=admnc.getLastGradientLogs().map { x => x.toInt.formatted("%d") }.mkString("|")
                  if (admnc.getLastGradientLogs()(0)==Double.MinValue)
                    convergenceInfo="CONVERGED"
                  
                  pw.println("\tFold AUROC:"+foldAuroc+" "+convergenceInfo)
                  pw.flush()
                  println("\tFold AUROC:"+foldAuroc+" "+convergenceInfo)

                  modelAUROC += foldAuroc
              }
              modelAUROC=modelAUROC/folds.length.toDouble
              println("\tAUROC:"+modelAUROC)
              if (modelAUROC>bestResult)
              {
                bestResult=modelAUROC
                bestR=regP
                bestL0=l0
                bestLS=ls
                bestG=g
                bestSD=sD
                bestLL=ll
                bestNR=nr
              }
              if (modelAUROC<worstResult)
              {
                worstResult=modelAUROC
                worstR=regP
                worstL0=l0
                worstLS=ls
                worstG=g
                worstSD=sD
                worstLL=ll
                worstNR=nr
              }
          }
    
    if (1.0-worstResult>bestResult)
    {
      bestR=worstR
      bestL0=worstL0
      bestLS=worstLS
      bestG=worstG
      bestSD=worstSD
      bestLL=worstLL
      bestNR=worstNR
      bestResult=worstResult
    }
    pw.println("BEST - R:"+bestR+" L0:"+bestL0+" LS:"+bestLS+" G:"+bestG+" K:"+bestSD+" LL:"+bestLL+" NR:"+bestNR+"\n\tobtained "+bestResult)
    println("BEST - R:"+bestR+" L0:"+bestL0+" LS:"+bestLS+" G:"+bestG+" K:"+bestSD+" LL:"+bestLL+" NR:"+bestNR+"\n\tobtained "+bestResult)
    val finishTime=System.currentTimeMillis()
    pw.println("Performed "+totalTests+" tests ("+folds.length+" folds) in "+((finishTime-startTime)/1000.0)+"s ("+((finishTime-startTime)/(1000.0*totalTests*folds.length))+"s per fold on average)")
    println("Performed "+totalTests+" tests ("+folds.length+" folds) in "+((finishTime-startTime)/1000.0)+"s ("+((finishTime-startTime)/(1000.0*totalTests*folds.length))+"s per fold on average)")
    pw.close
  }
}