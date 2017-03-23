package gal.udc

import scala.Ordering

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

object ADMNCModel
{
  val DEFAULT_SUBSPACE_DIMENSION:Int=2
  val DEFAULT_REGULARIZATION_PARAMETER:Double=0.01
  val DEFAULT_LEARNING_RATE_START:Double=1.0
  val DEFAULT_LEARNING_RATE_SPEED:Double=0.1
  val DEFAULT_FIRST_CONTINUOUS:Int=2
  val DEFAULT_MINIBATCH_SIZE:Int=100
  val DEFAULT_MAX_ITERATIONS:Int=50
  val DEFAULT_GAUSSIAN_COMPONENTS:Int=2
  val DEFAULT_NORMALIZING_R:Double=4
}

class ADMNCModel() extends Serializable
{
  //private var crf:CRFNoFactorModel=null
  private var crf:CRFModel=null
  private var gmm:GaussianMixtureModel=null
  private var threshold:Double=0-1
  var subspaceDimension=ADMNCModel.DEFAULT_SUBSPACE_DIMENSION
  var normalizingR=ADMNCModel.DEFAULT_NORMALIZING_R
  var maxIterations=ADMNCModel.DEFAULT_MAX_ITERATIONS
  var minibatchSize=ADMNCModel.DEFAULT_MINIBATCH_SIZE
  var regParameter=ADMNCModel.DEFAULT_REGULARIZATION_PARAMETER
  var learningRate0=ADMNCModel.DEFAULT_LEARNING_RATE_START
  var learningRateSpeed=ADMNCModel.DEFAULT_LEARNING_RATE_SPEED
  var gaussianK=ADMNCModel.DEFAULT_GAUSSIAN_COMPONENTS
  
  def trainWithSGD(sc:SparkContext, data:RDD[MixedData], anomalyRatio:Double)=
  {
    val numElems=data.count()
    //println("Training with "+numElems+" samples")
    val minibatchFraction=this.minibatchSize/numElems.toDouble
    val sampleElement=data.first()
    //this.crf=new CRFNoFactorModel(sampleElement.cPart.size, sampleElement.dPart.size, subspaceDimension)
    this.crf=new CRFModel(sampleElement.cPart.size, sampleElement.bPart.size, subspaceDimension, normalizingR)
    crf.trainWithSGD(sc, data, maxIterations, minibatchFraction, regParameter, learningRate0, learningRateSpeed)
    
    
    
    var tries=0;
    while(this.gmm==null) //For some **** reason this crashes from time to time
    {
      try
      {
        tries=tries+1
        this.gmm = new GaussianMixture().setK(gaussianK).run(data.map({x => Vectors.dense(x.cPart.toArray)}))
      }
      catch
      {
        case e:Exception =>  println("Gaussian mixture Model training failed!!")
        if (tries>5)
          System.exit(0)
      }
    }
    
    
    var estimators=getProbabilityEstimator(data)
    
    /* TODO - Search for the ith element effectively, not by sorting the whole dataset.
    var (minP, maxP)=probabilities.map({case (element, p) => (p, p)})
                                  .reduce({case ((max1, min1), (max2, min2)) => (Math.max(max1, max2),Math.min(min1, min2))})
    */                              
    var targetSize:Int=(numElems*anomalyRatio).toInt
    if (targetSize<=0) targetSize=1
    val anomalies=estimators.takeOrdered(targetSize)(Ordering[Double].on { x => x._2 })
    this.threshold=anomalies(anomalies.length-1)._2
    //println("Threshold: "+this.threshold)
  }
  def getProbabilityEstimator(element:MixedData):Double=
  {
    val crfEstimator=crf.getProbabilityEstimator(element)
    val gmmEstimator=gmm.gaussians.map({ g => g.pdf(Vectors.dense(element.cPart.toArray)) })
                                  .reduce({(a,b) => a + b /* OR IS IT MAX?*/})
    //return crfEstimator*gmmEstimator
    //Done like this in Matlab
    return Math.log(crfEstimator*gmmEstimator)
  }
  /* DEBUG
  def getGaussianEstimator(element:MixedData):Double=
  {
    return Math.log(gmm.gaussians.map({ g => g.pdf(Vectors.dense(element.cPart.toArray)) }).reduce({(a,b) => a + b /* OR IS IT MAX?*/}))
  }
  def getCRFProbabilityEstimator(element:MixedData):Double=
  {
    val crfEstimator=crf.getProbabilityEstimator(element)
    //return crfEstimator*gmmEstimator
    //Done like this in Matlab
    return Math.log(crfEstimator)
  }*/
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
}