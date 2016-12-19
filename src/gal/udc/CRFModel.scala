package gal.udc

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import breeze.linalg.DenseMatrix
import breeze.linalg.{ DenseVector => BDV }
import breeze.linalg.InjectNumericOps
import breeze.linalg.sum


class CRFModel(numContinuous:Int, numDiscrete:Int, subspaceDimension:Int, normalizingR:Double = 4) extends Serializable
{
  var Wxy:DenseMatrix[Double]=(DenseMatrix.rand(subspaceDimension, numContinuous+1) :- 0.5)//Rand can be specified - Subtract 0.5 because we want both positive and negative numbers.
  var Vxy:DenseMatrix[Double]=(DenseMatrix.rand(subspaceDimension, numDiscrete+1) :- 0.5)//Rand can be specified
  var Wyy:DenseMatrix[Double]=(DenseMatrix.rand(subspaceDimension, numDiscrete+1) :- 0.5)//Rand can be specified
  var Vyy:DenseMatrix[Double]=(DenseMatrix.rand(subspaceDimension, numDiscrete+1) :- 0.5)//Rand can be specified
  val NORMALIZING_R=normalizingR

  private def _getFactor(element:MixedData):Double=
  {
    val xCont=element.cPart
    val yDisc=element.dPart
    val w=(Wxy*xCont).dot(Vxy*yDisc) + (Wyy*yDisc).dot(Vyy*yDisc)
    return Math.exp(w)
  }
  private def _estimatePartitionFunction(element:MixedData):Double=
  {
    val sampleSize=1000
    val sample=GibbsSampler.obtainSampleWithFixedContinuousPart(sampleSize, element, _getFactor, true)
    //val sample=GibbsSampler.obtainSampleWithFixedContinuousPart(sampleSize, element, _getFactor)
    var z:Double=0
    sample.foreach({ x => z+=_getFactor(x) })
    val numCombinations=scala.math.pow(2,(element.dPart.length-1))//Not counting the bias.
    return z*numCombinations/sampleSize
  }
  def getProbabilityEstimator(element:MixedData):Double=
  {
    var adaptedElement=new MixedData(BDV.vertcat(element.cPart,BDV(1.0)),
                                     BDV.vertcat(element.dPart,BDV(1.0)))
    val f=_getFactor(adaptedElement)
    val z=_estimatePartitionFunction(adaptedElement)//, sampler)
    return f/z
  }
  def update(gradientWxy:DenseMatrix[Double], gradientVxy:DenseMatrix[Double], gradientWyy:DenseMatrix[Double], gradientVyy:DenseMatrix[Double], learningRate:Double, regularizationParameter:Double)
  {
    Wxy=Wxy+learningRate*(gradientWxy - 2*regularizationParameter*Wxy)
    Vxy=Vxy+learningRate*(gradientVxy - 2*regularizationParameter*Vxy)
    Wyy=Wyy+learningRate*(gradientWyy - 2*regularizationParameter*Wyy)
    Vyy=Vyy+learningRate*(gradientVyy - 2*regularizationParameter*Vyy)
    
    //Normalize cols
    Wxy=_normalizeColumns(Wxy)
    Vxy=_normalizeColumns(Vxy)
    Wyy=_normalizeColumns(Wyy)
    Vyy=_normalizeColumns(Vyy)
  }
  private def _normalizeColumns(m:DenseMatrix[Double]):DenseMatrix[Double]=
  {
    for (i <- 0 until m.cols)
    {
      var total=0.0
      for (j <- 0 until m.rows)
        total += m(j,i)*m(j,i)
      total=Math.sqrt(total)
      if (total>NORMALIZING_R)
        for (j <- 0 until m.rows)
          m(j,i)=m(j,i)*NORMALIZING_R/total
    }
    return m
  }
  def getPartialWxy(element:MixedData, sample:Array[MixedData]):DenseMatrix[Double]=
  {
    val xCont=element.cPart
    val totalNum:Double=sample.length
    var totalGradient=DenseMatrix.zeros[Double](Wxy.rows, Wxy.cols)
    //Map & reduce instead?
    sample.foreach({ e => totalGradient=totalGradient:+ (xCont.asDenseMatrix.t*(Vxy*e.dPart).asDenseMatrix).t})
    
    return (xCont.asDenseMatrix.t*(Vxy*element.dPart).asDenseMatrix).t :- (totalGradient :/ totalNum)
  }
  def getPartialVxy(element:MixedData, sample:Array[MixedData]):DenseMatrix[Double]=
  {
    val xCont=element.cPart
    val totalNum:Double=sample.length
    var totalGradient=DenseMatrix.zeros[Double](Vxy.rows, Vxy.cols)
    sample.foreach({ e => totalGradient=totalGradient:+ (e.dPart.asDenseMatrix.t*(Wxy*xCont).asDenseMatrix).t})
    return (element.dPart.asDenseMatrix.t*(Wxy*xCont).asDenseMatrix).t :- (totalGradient :/ totalNum)
  }
  def getPartialWyy(element:MixedData, sample:Array[MixedData]):DenseMatrix[Double]=
  {
    val xCont=element.cPart
    val totalNum:Double=sample.length
    var totalGradient=DenseMatrix.zeros[Double](Wyy.rows, Wyy.cols)
    sample.foreach({ e => totalGradient=totalGradient:+ (e.dPart.asDenseMatrix.t*(Vyy*e.dPart).asDenseMatrix).t})
    return (element.dPart.asDenseMatrix.t*(Vyy*element.dPart).asDenseMatrix).t :- (totalGradient :/ totalNum)
  }
  def getPartialVyy(element:MixedData, sample:Array[MixedData]):DenseMatrix[Double]=
  {
    val xCont=element.cPart
    val totalNum:Double=sample.length
    var totalGradient=DenseMatrix.zeros[Double](Vyy.rows, Vyy.cols)
    sample.foreach({ e => totalGradient=totalGradient:+ (e.dPart.asDenseMatrix.t*(Wyy*e.dPart).asDenseMatrix).t})
    return (element.dPart.asDenseMatrix.t*(Wyy*element.dPart).asDenseMatrix).t :- (totalGradient :/ totalNum)
  }
  def trainWithSGD(sc:SparkContext, data:RDD[MixedData], maxIterations:Int, minibatchFraction:Double, regParameter:Double, learningRate0:Double, learningRateSpeed:Double)=
  {
    val adaptedData=data.map({e => new MixedData(BDV.vertcat(e.cPart,BDV(1.0)), BDV.vertcat(e.dPart,BDV(1.0)))}) //Append 1's to vectors
    //SGD
    var consecutiveNoProgressSteps=0
    var i=1
    while ((i<maxIterations) && (consecutiveNoProgressSteps<10)) //Finishing condition: gradients are small for several iterations in a row
    {
      if (i%10==0)
        println("Starting iteration "+i)
      //Use a minibatch instead of the whole dataset
      var minibatch=adaptedData.sample(false, minibatchFraction, java.lang.System.nanoTime().toInt) //Sample the original set
      var total:Double=minibatch.count()
      while(total<=0)
      {
        minibatch=data.sample(false, minibatchFraction, java.lang.System.nanoTime().toInt) //Sample the original set
        total=minibatch.count()
      }
      
      val learningRate=learningRate0/(1+learningRateSpeed*(i-1))
      
      val (sumWxy, sumVxy, sumWyy, sumVyy)=
        minibatch.map({e => //val vSample=bSample.value
                            val vSample=GibbsSampler.obtainSampleWithFixedContinuousPart(1000,e, _getFactor, true) //Last variable should be ignored (it's the bias, must be 1).
                              //println("EJEMPLO:"+e)
                              (getPartialWxy(e, vSample),
                              getPartialVxy(e, vSample),
                              getPartialWyy(e, vSample),
                              getPartialVyy(e, vSample)
                              )})
                 .reduce({ (e1,e2) => (e1._1+e2._1,e1._2+e2._2,e1._3+e2._3,e1._4+e2._4)})
                 
      val gradientWxy=sumWxy:/total
      val gradientProgress=Math.abs(sum(gradientWxy))
      if (gradientProgress<0.00001)
        consecutiveNoProgressSteps=consecutiveNoProgressSteps+1
      else
        consecutiveNoProgressSteps=0
      
      update(gradientWxy, sumVxy:/total, sumWyy:/total, sumVyy:/total, learningRate, regParameter)
      i=i+1
    }
  }
  def print()=
  {
    println("Wxy:"+Wxy)
    println("Vxy:"+Vxy)
    println("Wyy:"+Wyy)
    println("Vyy:"+Vyy)
  }
}