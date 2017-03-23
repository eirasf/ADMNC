package gal.udc

import breeze.linalg.{DenseVector => BDV}
import scala.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{DenseVector => BDV}

object MixedData
{
  def fromLabeledPointRDD(lpRDD:RDD[LabeledPoint], firstContinuous:Int, normalizeContinuous:Boolean):RDD[(MixedData,Boolean)]=
  {
    if (!normalizeContinuous)
    {
      val maxs:Array[Int]=lpRDD.map({ x => x.features.toArray.slice(0, firstContinuous) })
                                  .reduce({ (x,y) =>x.zip(y).map({case (a,b) => Math.max(a,b)})})
                                  .map({ x => x.toInt})
      return lpRDD.map({ x => (new MixedData(BDV(x.features.toArray.slice(firstContinuous, x.features.size)),
                                             BDV(x.features.toArray.slice(0, firstContinuous)),
                                             maxs),
                               x.label==1) })
    }
    else
    {
      val maxs:(Array[Double],Array[Double],Array[Double])=lpRDD.map({ x => var cont=x.features.toArray.slice(firstContinuous, x.features.size)
                                                                    (x.features.toArray.slice(0, firstContinuous), cont, cont) })
                                                         .reduce({ (x,y) => (x._1.zip(y._1).map({case (a,b) => Math.max(a,b)}),
                                                                             x._2.zip(y._2).map({case (a,b) => Math.max(a,b)}),
                                                                             x._3.zip(y._3).map({case (a,b) => Math.min(a,b)}))})
      val maxsDisc:Array[Int]=maxs._1.map(_.toInt)
      val limitsCont:Array[(Double,Double)]=maxs._2.zip(maxs._3).map({case (x, y) => (y,x - y)})
      return lpRDD.map({ x => var cont=x.features.toArray.slice(firstContinuous, x.features.size)
                              cont=cont.zip(limitsCont).map({case (x,(m,r)) => (x-m)/(r+0.0000000000001)})
                              (new MixedData(BDV(cont),
                                             BDV(x.features.toArray.slice(0, firstContinuous)),
                                             maxsDisc),
                               x.label==1) })
    }
  }
}

@SerialVersionUID(100L)
class MixedData(continuousPart:BDV[Double], discretePart:BDV[Double], numValuesForDiscretes:Array[Int]=null) extends Serializable
{
  val cPart:BDV[Double]=continuousPart
  var _dPart:BDV[Double]=discretePart
  def dPart=_dPart
  var _modified=true
  def setDPart(index:Int, value:Double)=
    {
      _modified=true
      _dPart(index)=value
    }
  var isArtificialAnomaly:Boolean=false
  val nValuesForDiscretes:Array[Int]=numValuesForDiscretes
  def bPart:BDV[Double]=nValuesForDiscretes match //Indicator variables with a bias
                        {
                          case null => if (_modified)
                                       {
                                          _cachedBPart=BDV.vertcat(_dPart,BDV(1.0))
                                          _modified=false
                                       }
                                       _cachedBPart
                          case anything => if (_modified)
                                           {
                                              _cachedBPart=BDV.vertcat(OneHotEncode(_dPart, anything),BDV(1.0))
                                              _modified=false
                                           }
                                           _cachedBPart
                        }
  //Cache bPart
  var _cachedBPart:BDV[Double]=null
  
  val numDiscreteCombinations:Double=nValuesForDiscretes match
                        {
                          case null => scala.math.pow(2,(bPart.length-1))
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
      
      if (current+d(i).toInt>=binarizedValues.length)
        i=0;
      binarizedValues(current+d(i).toInt)=1
      
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
  
  override def toString():String=
  {
    (if (isArtificialAnomaly) "ANOMALY - " else "")+"C:"+cPart.toString()+"##D:"+dPart.toString()
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
    }
  }
}