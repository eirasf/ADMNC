package gal.udc

import scala.Ordering

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

import breeze.linalg.{ DenseMatrix => BDM }
import breeze.linalg.{ DenseVector => BDV }
import breeze.linalg.{ Vector => BV }
import breeze.linalg.diag
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

object ADMNC_LogisticModel
{
  val DEFAULT_SUBSPACE_DIMENSION:Int=10
  val DEFAULT_REGULARIZATION_PARAMETER:Double=1
  val DEFAULT_LEARNING_RATE_START:Double=1.0
  val DEFAULT_LEARNING_RATE_SPEED:Double=0.1
  val DEFAULT_FIRST_CONTINUOUS:Int=2
  val DEFAULT_MINIBATCH_SIZE:Int=100
  val DEFAULT_MAX_ITERATIONS:Int=100
  val DEFAULT_GAUSSIAN_COMPONENTS:Int=4
  val DEFAULT_NORMALIZING_R:Double=10
  val DEFAULT_LOGISTIC_LAMBDA:Double=1
}

class ADMNC_LogisticModel() extends Serializable
{
  //private var crf:CRFNoFactorModel=null
  private var logistic:LogisticModel=null
  private var gmm:GaussianMixtureModel=null
  private var threshold:Double=0-1
  var maxIterations=ADMNC_LogisticModel.DEFAULT_MAX_ITERATIONS
  var minibatchSize=ADMNC_LogisticModel.DEFAULT_MINIBATCH_SIZE
  var regParameter=ADMNC_LogisticModel.DEFAULT_REGULARIZATION_PARAMETER
  var learningRate0=ADMNC_LogisticModel.DEFAULT_LEARNING_RATE_START
  var learningRateSpeed=ADMNC_LogisticModel.DEFAULT_LEARNING_RATE_SPEED
  var gaussianK=ADMNC_LogisticModel.DEFAULT_GAUSSIAN_COMPONENTS
  var logisticLambda=ADMNC_LogisticModel.DEFAULT_LOGISTIC_LAMBDA
  
  def toBreeze(v:Vector) = BV(v.toArray)
  def fromBreeze(bv:BV[Double]) = Vectors.dense(bv.toArray)
  def fromBreeze(bm:BDM[Double]) = new DenseMatrix(bm.rows, bm.cols, bm.toArray)
  def add(v1:Vector, v2:Vector) = fromBreeze( toBreeze(v1) + toBreeze(v2))
  def elementWiseSqrt(bv:BV[Double]):BV[Double]=
  {
    val elems=bv.toArray
    for (i <- 0 until elems.length)
      elems(i)=Math.sqrt(elems(i))
    return BDV(elems)
  }
  
  def trainWithSGD(sc:SparkContext, data:RDD[MixedData], anomalyRatio:Double)=//TODO - Have optional parameter to avoid transformation of data on multiple tests with same data
  {
    //val transformedData=data.flatMap { x => x.toMaskedRepresentation() }
    val sampleElement=data.first()
    val numElems=data.count()
    var minibatchFraction=this.minibatchSize/numElems.toDouble
    if (minibatchFraction>1)
      minibatchFraction=1
    
    
    this.logistic=new LogisticModel(sampleElement.nPart.size, sampleElement.getMaskedRepresentationElement(0).mPart.size, logisticLambda)
    logistic.trainWithSGD(sc, data, maxIterations, minibatchFraction, regParameter, learningRate0, learningRateSpeed)
    
    
    
    var tries=0;
    //We want to have at least enough elements to have 1 in each cluster. We don't want clusters for elements with probability < anomalyRatio
    val minElems=gaussianK*(Math.log(0.5)/Math.log(1-anomalyRatio))*10//Heuristic
    //println(minElems)
    
    val allDataRDD=data.map({x => Vectors.dense(x.nPart.toArray)})
    val nElems=data.count()
    val dataRDD=if (nElems<=minElems)
                  allDataRDD
                else
                  allDataRDD.sample(false, minElems.toDouble/nElems, System.nanoTime())
    dataRDD.cache()
    while(this.gmm==null) //For some reason this crashes from time to time. Improved by initializing with KMeans.
    {
      try
      {
        tries=tries+1
        val clusters = KMeans.train(dataRDD, gaussianK, 10)
    
        val clStdDevs = dataRDD.map({ case x =>
                                            val predicted=clusters.predict(x)
                                            val d=toBreeze(x)-toBreeze(clusters.clusterCenters(predicted))
                                            (predicted,(d:*d,1)) })
                                .reduceByKey({ case (x,y) => (x._1:+y._1, x._2+y._2)})
                                .map({case (cl,(s,c)) => (cl, elementWiseSqrt(s).:/(c.toDouble)) })
        //clStdDevs.foreach(println)
        val st=clStdDevs.first()._2
        val gaussianInits=clStdDevs.map({case (c, stddevV) =>
                                              new MultivariateGaussian(clusters.clusterCenters(c), fromBreeze(diag(stddevV.toDenseVector)))})
                                   .collect()
        val initialWeights=Array.fill(gaussianInits.length)(1.0/gaussianInits.length)
        val initialModel=new GaussianMixtureModel(initialWeights, gaussianInits)
        this.gmm = new GaussianMixture().setK(gaussianInits.length).setInitialModel(initialModel).run(dataRDD)
        
        //OLD VERSION
        //this.gmm = new GaussianMixture().setK(gaussianK).run(data.map({x => Vectors.dense(x.cPart.toArray)}))
      }
      catch
      {
        case e:Exception =>  println("Gaussian mixture Model training failed!!")
        if (tries>20)
          System.exit(0)
      }
    }
    
    
    var estimators=getProbabilityEstimator(data)
    
    //estimators.take(10).foreach(println)
    
    /* TODO - Search for the ith element effectively, not by sorting the whole dataset.
    var (minP, maxP)=probabilities.map({case (element, p) => (p, p)})
                                  .reduce({case ((max1, min1), (max2, min2)) => (Math.max(max1, max2),Math.min(min1, min2))})
    */                              
    var targetSize:Int=(numElems*anomalyRatio).toInt
    if (targetSize<=0) targetSize=1
//    val anomalies=estimators.takeOrdered(targetSize)(Ordering[Double].on { x => x._2 })
//    this.threshold=anomalies(anomalies.length-1)._2
//TEMP - To speed up grid
this.threshold=0.1
    //println("Threshold: "+this.threshold)
  }
  def getProbabilityEstimator(element:MixedData):Double=
  {
    val logisticEstimator=logistic.getProbabilityEstimator(element)
    val gmmEstimator=gmm.gaussians.map({ g => g.pdf(Vectors.dense(element.nPart.toArray)) })
                                  .reduce({(a,b) => a + b /* OR IS IT MAX?*/})
    //return crfEstimator*gmmEstimator
    //Done like this in Matlab
    return Math.log(logisticEstimator*gmmEstimator)
  }
  /* DEBUG */
  def getGaussianEstimator(element:MixedData):Double=
  {
    return Math.log(gmm.gaussians.map({ g => g.pdf(Vectors.dense(element.nPart.toArray)) }).reduce({(a,b) => a + b /* OR IS IT MAX?*/}))
  }
  def getLogisticProbabilityEstimator(element:MixedData):Double=
  {
    val logisticEstimator=logistic.getProbabilityEstimator(element)
    //return crfEstimator*gmmEstimator
    //Done like this in Matlab
    return Math.log(logisticEstimator)
  }
  def getProbabilityEstimator(data:RDD[MixedData]):RDD[(MixedData,Double)]=
  {
    return data.map({ x => (x,getProbabilityEstimator(x))})
  }
  def isAnomaly(element:MixedData):Boolean=
  {
    return getProbabilityEstimator(element) < this.threshold
  }
  def isAnomaly(data:RDD[(MixedData)]):RDD[(MixedData,Boolean)]=
  {
    return data.map({ x => (x,isAnomaly(x))})
  }
  def getLastGradientLogs():Array[Double]=
  {
    if (logistic==null) return LogisticModel.UNCONVERGED_GRADIENTS.clone()
    return logistic.lastGradientLogs
  }
}