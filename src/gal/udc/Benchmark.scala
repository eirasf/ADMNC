package gal.udc

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions

import breeze.linalg.{ DenseVector => BDV }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import java.io.PrintWriter
import java.io.File
import java.util.Calendar


object Benchmark
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
                                                    "output_file" -> DEFAULT_OUTPUT_FILE,
                                                    "start" -> null)
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
          case "s"  => "start"
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
        if (option=="start")
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
    val firstContinuous=options("first_continuous").asInstanceOf[Double].toInt - 1
    val dataRDD=MixedData.fromLabeledPointRDD(data, firstContinuous, true)
    //val dataRDD=MixedData.fromLabeledPointRDD(data, firstContinuous, false)
    
    val sampleElement=dataRDD.first
    println("Sample element:"+sampleElement)
    
    dataRDD.cache()
    
    //val folds=MLUtils.kFold(dataRDD, 10, System.nanoTime().toInt)
    val folds=MLUtils.kFold(dataRDD, 5, System.nanoTime().toInt)
    
    //val regPars=Array[Double](0.01, 0.1, 1.0, 10)
    val regPars=Array[Double](0.01, 0.1, 1.0, 10)
    //val estimatedPercentages=Array[Double](0.01, 0.03)
    //val subspaceDimensions=Array[Int](2, 4, 6)
    val subspaceDimensions=Array[Int](2)
    //val numGaussians=Array[Int](2, 4, 6)
    val numGaussians=Array[Int](2)
    //val lambdas=Array[Double](0.01, 0.1, 1, 10, 100)
    val lambdas=Array[Double](0.01, 1, 100)
    //val lambdaSpeeds=Array[Double](0.001, 0.01, 0.1, 1, 10, 100)
    val lambdaSpeeds=Array[Double](0.001, 0.1, 10)
    //val normalizingRs=Array[Double](0.1, 1.0)
    val normalizingRs=Array[Double](0.1, 10)
    
    //var (bestResult, bestR, bestP, bestSD, bestNG) = (1.0, regPars(0), estimatedPercentages(0), subspaceDimensions(0), numGaussians(0))
    var (bestResult, bestR, bestL0, bestLS, bestG, bestSD, bestNR) = (1.0, regPars(0), lambdas(0), lambdaSpeeds(0), numGaussians(0), subspaceDimensions(0), normalizingRs(0))
    var (worstResult, worstR, worstL0, worstLS, worstG, worstSD, worstNR) = (0.0, regPars(0), lambdas(0), lambdaSpeeds(0), numGaussians(0), subspaceDimensions(0), normalizingRs(0))
    var (bestAUROCResult, bestAUROCR, bestAUROCL0, bestAUROCLS, bestAUROCG, bestAUROCSD, bestAUROCNR) = (0.0, regPars(0), lambdas(0), lambdaSpeeds(0), numGaussians(0), subspaceDimensions(0), normalizingRs(0))
    var (worstAUROCResult, worstAUROCR, worstAUROCL0, worstAUROCLS, worstAUROCG, worstAUROCSD, worstAUROCNR) = (1.0, regPars(0), lambdas(0), lambdaSpeeds(0), numGaussians(0), subspaceDimensions(0), normalizingRs(0))
    
    val pw = new PrintWriter(new File(options("output_file").asInstanceOf[String]))
    pw.println(options("dataset").asInstanceOf[String]+" "+Calendar.getInstance().getTime())
    
    var startingIndices=options("start").asInstanceOf[String]
    
    if (startingIndices==null)
      startingIndices="0/0/0/0/0/0";
    
    var startingIndicesA=startingIndices.split("/");
    
    var fnr=startingIndicesA(0).toInt;
    var fg=startingIndicesA(1).toInt;
    var fs=startingIndicesA(2).toInt;
    var fr=startingIndicesA(3).toInt;
    var fl=startingIndicesA(4).toInt;
    var fls=startingIndicesA(5).toInt;
    
    var first=true;
    
    var counter=0
    val total=regPars.length*subspaceDimensions.length*numGaussians.length*lambdas.length*lambdaSpeeds.length*normalizingRs.length
    
    //TODO - Do this the Spark 2.0 way: with Dataset, ParamGrid and such,
    //for (nr <- normalizingRs)
    for (inr <- fnr until normalizingRs.length)
    {
      var nr=normalizingRs(inr)
      var fig=if (first) fg else 0
      //for (g <- numGaussians)
      for (ig <- fig until numGaussians.length)
      {
        var g=numGaussians(ig)
        var fiSD=if (first) fs else 0
        //for (sD <- subspaceDimensions)
        for (isD <- fiSD until subspaceDimensions.length)
        {
          var sD=subspaceDimensions(isD)
          var fir=if (first) fr else 0
          //for (regP <- regPars)
          for (iregP <- fir until regPars.length)
          {
            var regP=regPars(iregP)
            var fil0=if (first) fl else 0
            //for (l0 <- lambdas)
            for (il0 <- fil0 until lambdas.length)
            {
              var l0=lambdas(il0)
              var fils=if (first) fls else 0
              //for (ls <- lambdaSpeeds)
              for (ils <- fils until lambdaSpeeds.length)
              {
                counter=counter+1
                var ls=lambdaSpeeds(ils)
                first=false
                pw.println("["+counter+"/"+total+"] NR:"+nr+" G:"+g+" K:"+sD+" R:"+regP+" L0:"+l0+" LS:"+ls)
                println("["+counter+"/"+total+"] NR:"+nr+" G:"+g+" K:"+sD+" R:"+regP+" L0:"+l0+" LS:"+ls)
                var modelMCR=0.0
                var modelAUROC=0.0
                var fCount=0
                for (fold <- folds)
                {
                  fCount=fCount+1
                  //Model configuration, creation and training
                  val admcc=new ADMNCModel()
                  admcc.subspaceDimension=sD
                  admcc.maxIterations=options("max_iterations").asInstanceOf[Double].toInt
                  admcc.minibatchSize=options("minibatch").asInstanceOf[Double].toInt
                  admcc.regParameter=regP
                  admcc.learningRate0=l0
                  admcc.learningRateSpeed=ls
                  admcc.gaussianK=g
                  
                  admcc.normalizingR=nr
                  
                  //admcc.trainWithSGD(sc, fold._1.map(_._1), perc)
                  try
                  {
                    admcc.trainWithSGD(sc, fold._1.filter(!_._2).map(_._1), 0.01)
                    
                    //var probabilityRDD=admcc.getProbabilityEstimator(dataRDD.map(_._1))
                    var misclassificationRate=fold._2.map({case (element, label) => val result=admcc.isAnomaly(element)
                                                                   if (result==label)
                                                                     0.0
                                                                   else
                                                                     1.0
                                                                   }).sum()/fold._2.count()
                    modelMCR=modelMCR+misclassificationRate
                    pw.println("Fold "+fCount+"/"+folds.length+" MCR:"+misclassificationRate);
                    println("Fold "+fCount+"/"+folds.length+" MCR:"+misclassificationRate);
                    
                    val probs=fold._2.map({case (element, label) =>(admcc.getProbabilityEstimator(element), if (label) 0.0 else 1.0)
                                                                               })//Mark as positive non-anomalous elements, which are given high estimators.
                    val metrics = new BinaryClassificationMetrics(probs)
                    // ROC Curve
                    val roc = metrics.roc
                    
                    // AUROC
                    val auROC = metrics.areaUnderROC
                    modelAUROC=modelAUROC+auROC
                    pw.println("Fold "+fCount+"/"+folds.length+" AUROC:"+auROC);
                    println("Fold "+fCount+"/"+folds.length+" AUROC:"+auROC);
                  }catch
                  {
                    case e:Exception => modelMCR=modelMCR+1
                                        modelAUROC=modelAUROC+0.5
                                        pw.println("Fold crashed!!");
                                        println("Fold crashed!!");
                  }
                }
                modelMCR=modelMCR/folds.length.toDouble
                pw.println("MCR:"+modelMCR)
                pw.flush()
                if (modelMCR<bestResult)
                {
                  bestResult=modelMCR
                  bestR=regP
                  bestL0=l0
                  bestLS=ls
                  bestG=g
                  bestSD=sD
                  bestNR=nr
                }
                if (modelMCR>worstResult)
                {
                  worstResult=modelMCR
                  worstR=regP
                  worstL0=l0
                  worstLS=ls
                  worstG=g
                  worstSD=sD
                  worstNR=nr
                }
                
                modelAUROC=modelAUROC/folds.length.toDouble
                pw.println("AUROC:"+modelAUROC)
                pw.flush()
                if (modelAUROC>bestAUROCResult)
                {
                  bestAUROCResult=modelAUROC
                  bestAUROCR=regP
                  bestAUROCL0=l0
                  bestAUROCLS=ls
                  bestAUROCG=g
                  bestAUROCSD=sD
                  bestAUROCNR=nr
                }
                if (modelAUROC<worstAUROCResult)
                {
                  worstAUROCResult=modelAUROC
                  worstAUROCR=regP
                  worstAUROCL0=l0
                  worstAUROCLS=ls
                  worstAUROCG=g
                  worstAUROCSD=sD
                  worstAUROCNR=nr
                }
              }
            }
          }
        }
      }
    }
    pw.println("BEST MCR - R:"+bestR+" L0:"+bestL0+" LS:"+bestLS+" G:"+bestG+" K:"+bestSD+" NR:"+bestNR+" obtained "+bestResult+" MCR")
    pw.println("WORST MCR - R:"+worstR+" L0:"+worstL0+" LS:"+worstLS+" G:"+worstG+" K:"+worstSD+" NR:"+worstNR+" obtained "+worstResult+" MCR")
    pw.println("BEST AUROC - R:"+bestAUROCR+" L0:"+bestAUROCL0+" LS:"+bestAUROCLS+" G:"+bestAUROCG+" K:"+bestAUROCSD+" NR:"+bestAUROCNR+" obtained "+bestAUROCResult+" AUROC")
    pw.println("WORST AUROC - R:"+worstAUROCR+" L0:"+worstAUROCL0+" LS:"+worstAUROCLS+" G:"+worstAUROCG+" K:"+worstAUROCSD+" NR:"+worstAUROCNR+" obtained "+worstAUROCResult+" AUROC")
    pw.close
  }
}
