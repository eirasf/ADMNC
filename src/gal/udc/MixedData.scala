package gal.udc

import scala.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.DenseVector
import scala.util.control.Breaks._

object MixedData
{
  def fromLabeledPointRDD(lpRDD:RDD[LabeledPoint], firstContinuous:Int, normalizeContinuous:Boolean, addBias:Boolean=true):RDD[(MixedData,Boolean)]=
  {
    if (!normalizeContinuous)
    {
      val maxs:Array[Int]=lpRDD.map({ x => x.features.toArray.slice(0, firstContinuous) })
                                  .reduce({ (x,y) =>x.zip(y).map({case (a,b) => Math.max(a,b)})})
                                  //.map({ x => if (x>1) x.toInt + 1 else 1})
                                  .map({ x => x.toInt})
      return lpRDD.map({ x => (new MixedData(BDV(x.features.toArray.slice(firstContinuous, x.features.size)),
                                             BDV(x.features.toArray.slice(0, firstContinuous)),
                                             maxs,
                                             addBias),
                               x.label==1) })
    }
    else
    {
      val maxs:(Array[Double],Array[Double],Array[Double])=lpRDD.map({ x => var cont=x.features.toArray.slice(firstContinuous, x.features.size)
                                                                    (x.features.toArray.slice(0, firstContinuous), cont, cont) })
                                                         .reduce({ (x,y) => (x._1.zip(y._1).map({case (a,b) => Math.max(a,b)}),
                                                                             x._2.zip(y._2).map({case (a,b) => Math.max(a,b)}),
                                                                             x._3.zip(y._3).map({case (a,b) => Math.min(a,b)}))})
      //val maxsDisc:Array[Int]=maxs._1.map({ x => if (x>1) x.toInt +1 else 1})
      val maxsDisc:Array[Int]=maxs._1.map(_.toInt)
      val limitsCont:Array[(Double,Double)]=maxs._2.zip(maxs._3).map({case (x, y) => (y,x - y)})
      return lpRDD.map({ x => var cont=x.features.toArray.slice(firstContinuous, x.features.size)
                              cont=cont.zip(limitsCont).map({case (x,(m,r)) => (x-m)/(r+0.0000000000001)})
                              (new MixedData(BDV(cont),
                                             BDV(x.features.toArray.slice(0, firstContinuous)),
                                             maxsDisc,
                                             addBias),
                               x.label==1) })
    }
  }
}

@SerialVersionUID(100L)
class MixedData(numericalPart:BDV[Double], categoricalPart:BDV[Double], numValuesForDiscretes:Array[Int]=null, addBias:Boolean=true) extends Serializable
{
  val nPart:BDV[Double]=numericalPart
  /*var dPart:BDV[Double]=numValuesForDiscretes match
                        {
                          case null => discretePart
                          case anything => OneHotEncode(discretePart, anything) 
                        }*/
  var _cPart:BDV[Double]=categoricalPart
  def dPart=_cPart
  var _modified=true
  def setDPart(index:Int, value:Double)=
    {
      _modified=true
      _cPart(index)=value
    }
  var isArtificialAnomaly:Boolean=false
  val nValuesForDiscretes:Array[Int]=numValuesForDiscretes
  def binarizedCategoricalPart:BDV[Double]=nValuesForDiscretes match //Indicator variables with a bias
                        {
                          case null => if (_modified)
                                       {
                                          if (addBias)
                                            _cachedBPart=BDV.vertcat(_cPart,BDV(1.0))
                                           else
                                             _cachedBPart=_cPart
                                          _modified=false
                                       }
                                       _cachedBPart
                          case anything => if (_modified)
                                           {
                                              if (addBias)
                                                _cachedBPart=BDV.vertcat(OneHotEncode(_cPart, anything),BDV(1.0))
                                              else
                                                _cachedBPart=OneHotEncode(_cPart, anything)
                                              _modified=false
                                           }
                                           _cachedBPart
                        }
  //Cache bPart
  var _cachedBPart:BDV[Double]=null
  
  val numDiscreteCombinations:Double=nValuesForDiscretes match
                        {
                          case null => scala.math.pow(2,(binarizedCategoricalPart.length-1))
                          case anything => var total:Double=1
                                            for (i <- 0 until anything.length)
                                              total = total * (anything(i)+1)
                                            total
                        }
  
  
  def OneHotEncode(d:BDV[Double], maxValues:Array[Int]):BDV[Double]=
  {
    val binarizedValues=Array.fill[Double](maxValues.sum + maxValues.length)(0)
    var i=0
    var current=0
    while (i<d.length)
    {
      //if (maxValues(i)>0)
      //{
        if (current+d(i).toInt>=binarizedValues.length)
          i=0;
        binarizedValues(current+d(i).toInt)=1
      //}
      //else
      //  binarizedValues(current)=d(i)
      current=current+(maxValues(i)+1)
      i=i+1
    }
    return BDV(binarizedValues)
  }
  def makeAnomaly(totalOptions:Int):MixedData=
  {
    isArtificialAnomaly=true
    val current=dPart(0)
    var newValue=current
    while(newValue==current)
      newValue=scala.util.Random.nextInt(totalOptions)
    setDPart(0,newValue)
    return this
  }
  
  def makeAnomaly(noiseSigma:Double, numNoiseVariables:Int):MixedData=
  {
    isArtificialAnomaly=true
    
    var nNoiseVars=numNoiseVariables
    if (numNoiseVariables>=nPart.length+_cPart.length)
      nNoiseVars=nPart.length+_cPart.length
    
    val fractionNoiseVariables=nNoiseVars.toDouble/(nPart.length+_cPart.length)
    var currentNoiseVariables=0
    var isNoise=new Array[Boolean](nPart.length+_cPart.length)
    
    while(currentNoiseVariables<numNoiseVariables)
    {
      for (i <- 0 until _cPart.length)
        if ((currentNoiseVariables<numNoiseVariables) && !isNoise(i) && (Math.random()<fractionNoiseVariables))
        {
          setDPart(i,(_cPart(i)+1)%2)
          currentNoiseVariables+=1
          isNoise(i)=true
        }
    
      val gRandom=new scala.util.Random()
      for (i <- 0 until nPart.length)
        if ((currentNoiseVariables<numNoiseVariables) && !isNoise(_cPart.length+i) && (Math.random()<fractionNoiseVariables))
        {
          nPart(i)+=gRandom.nextGaussian()*noiseSigma
          currentNoiseVariables+=1
          isNoise(_cPart.length+i)=true
        }
    }
    return this
  }
  
  override def toString():String=
  {
    (if (isArtificialAnomaly) "ANOMALY - " else "")+"C:"+nPart.toString()+"##D:"+dPart.toString()
  }
  
  def randomlySwitchDiscreteVariable(index:Int, calculateFactor:(MixedData) => Double)=
  {
    if(nValuesForDiscretes!=null)
    {
      val maxValue=nValuesForDiscretes(index)
      val thresholds=Array.fill[Double](maxValue+1)(0)
      var total:Double=0
      for (i <- 0 to maxValue)
      {
        setDPart(index,i)
        total=total+calculateFactor(this)
        thresholds(i)=total
      }
      val pick=Random.nextFloat*total
      var i=0
      var found=false
      while(!found && (i <= maxValue))
      {
        if (pick<thresholds(i))
        {
          setDPart(index,i)
          found=true
        }
        i=i+1;
      }
    }
    else
    {
      setDPart(index,0.0)
      val subfactor0=calculateFactor(this)
      setDPart(index,1.0)
      val subfactor1=calculateFactor(this)
      val threshold=subfactor0/(subfactor0+subfactor1)
      if (Random.nextFloat < threshold)
        setDPart(index,0.0)
      //else it already is 1.0
    }
  }
  
  def toMaskedRepresentation():Iterator[MaskRepresentationData]=
  {
    for (i <- Iterator.range(0, binarizedCategoricalPart.length-1)) yield getMaskedRepresentationElement(i)
    /*
    var result:Array[MaskRepresentationData] = new Array[MaskRepresentationData](bPart.length-1)
    
    for (i <- 0 until bPart.length-1)
    {
      val newMask=BDV.zeros[Double](bPart.length-1)
      newMask(i)=1.0
      result(i)=new MaskRepresentationData(cPart, newMask, bPart(i))
    }
    
    return result*/
  }
  
  def getMaskedRepresentationElement(index:Int):MaskRepresentationData=
  {
    val newMask=BDV.zeros[Double](binarizedCategoricalPart.length-1)
    newMask(index)=1.0
    return new MaskRepresentationData(nPart, newMask, binarizedCategoricalPart(index))
  }
  
  def getRandomMaskedRepresentationElement():MaskRepresentationData=
  {
    val rnd=new Random()
    val index=rnd.nextInt(binarizedCategoricalPart.length-1)
    return getMaskedRepresentationElement(index)
  }
}