package gal.udc

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import breeze.linalg.DenseMatrix
import breeze.linalg.{ DenseVector => BDV }
import breeze.linalg.InjectNumericOps
import breeze.linalg.sum
import scala.util.Random


object LogisticModel
{
  val UNCONVERGED_GRADIENTS=Array[Double](Double.MaxValue,Double.MaxValue,Double.MaxValue,Double.MaxValue,Double.MaxValue)
  val FULLY_CONVERGED_GRADIENTS=Array[Double](Double.MinValue,Double.MinValue,Double.MinValue,Double.MinValue,Double.MinValue)
}

class LogisticModel(numContinuous:Int, numDiscrete:Int, lambda:Double = 1) extends Serializable
{
  //TODO Names may be swapped
  var W:BDV[Double]=(BDV.rand(numContinuous+numDiscrete) :- 0.5)//Rand can be specified - Extra bias term added.
  
  val LAMBDA=lambda
  var regParameterScaleW:Double=1/Math.sqrt(numContinuous+numDiscrete)
  //val sampler=new GibbsSampler()
  var lastGradientLogs:Array[Double]=LogisticModel.UNCONVERGED_GRADIENTS.clone()
  
  def getProbabilityEstimator(element:MixedData):Double=
  {
    var prob=1.0
    
    val components=element.toMaskedRepresentation()
    val probs=components.map({ element => 
                                val yDisc=element.mPart
                                val xy=BDV.vertcat(element.cPart,yDisc)
                                val z=if (element.label==1.0) element.label else -1.0
                                val w=W.dot(xy)
                                val p=1.0/(1.0+Math.exp(-z*w/LAMBDA))
                                p})
    //probs.foreach { p => println("##"+p) }
              //.reduce(_*_)
    val probf=probs.reduce(_*_)
    //println(probf)
    return probf
  }
  def update(gradientW:BDV[Double], learningRate:Double, regularizationParameter:Double)
  {
    W=W-learningRate*(gradientW + 2*regularizationParameter*regParameterScaleW*W)
  }
  def trainWithSGD(sc:SparkContext, data:RDD[MixedData], maxIterations:Int, minibatchFraction:Double, regParameter:Double, learningRate0:Double, learningRateSpeed:Double)=
  {
    //SGD
    var consecutiveNoProgressSteps=0
    var i=1
    while ((i<maxIterations) && (consecutiveNoProgressSteps<10)) //Finishing condition: gradients are small for several iterations in a row
    {
      //Use a minibatch instead of the whole dataset
      var minibatch=data.sample(false, minibatchFraction, java.lang.System.nanoTime().toInt) //Sample the original set
      var total:Double=minibatch.count()
      while(total<=0)
      {
        minibatch=data.sample(false, minibatchFraction, java.lang.System.nanoTime().toInt) //Sample the original set
        total=minibatch.count()
      }
      
      val learningRate=learningRate0/(1+learningRateSpeed*(i-1))
      
      //Broadcast samples and parallelize on the minibatch
      //val bSample=sc.broadcast(sample)
      val (sumdW)=
        minibatch.map({e => val elem=e.getRandomMaskedRepresentationElement
                            val z=if (elem.label==1.0) elem.label else -1.0
                            val yDisc=elem.mPart
                            val xy=BDV.vertcat(elem.cPart,yDisc)
                            val w=W.dot(xy)
                            val s=1.0/(1.0+Math.exp(z*w/LAMBDA)) // Minus sigma(-x)
                              (-s*z*xy)})
                 .reduce({ (e1,e2) => (e1+e2)})
      /* TEST CODE - Only one example in the minibatch used on each step
      //println("DEBUG - Single item minibatch")
      val preW=W.toArray
      val preV=V.toArray
      total=1
      val e=minibatch.takeSample(false,1)(0)
      val z=if (e.label==1.0) e.label else -1.0
      val xCont=BDV.vertcat(e.cPart,BDV(1.0))//e.cPart
      val yDisc=e.mPart
      val w=(V*xCont).dot(W*yDisc)
      val s=1.0/(1.0+Math.exp(z*w/LAMBDA)) // Minus sigma(-x)
      val sumW = -s*z*(yDisc.asDenseMatrix.t*(V*xCont).asDenseMatrix).t
      val sumV = -s*z*(xCont.asDenseMatrix.t*(W*yDisc).asDenseMatrix).t
      */
      val gradientW=sumdW:/total
      //val gradientProgress=Math.abs(sum(gradientWxy))
      val gradientProgress=sum(gradientW.map({ x => Math.abs(x) }))
      if (gradientProgress<0.00001)
        consecutiveNoProgressSteps=consecutiveNoProgressSteps+1
      else
        consecutiveNoProgressSteps=0
        
      /* DEBUG
      if (i%10==0)
        println("Gradient size:"+gradientProgress)
      */
      //println("Gradient size:"+gradientProgress)
      
      if (i>=maxIterations-lastGradientLogs.length)
        lastGradientLogs(i-maxIterations+lastGradientLogs.length)=Math.log(gradientProgress)
        
      update(gradientW, learningRate, regParameter)
      
      i=i+1
    }
    if (consecutiveNoProgressSteps>=10)
      lastGradientLogs=LogisticModel.FULLY_CONVERGED_GRADIENTS
  }
  def print()=
  {
    println("W:"+W)
  }
}