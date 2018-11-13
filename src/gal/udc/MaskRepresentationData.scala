package gal.udc

import breeze.linalg.{DenseVector => BDV}
import scala.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{DenseVector => BDV}


@SerialVersionUID(100L)
class MaskRepresentationData(continuousPart:BDV[Double], mask:BDV[Double], labelValue:Double) extends Serializable
{
  val cPart:BDV[Double]=continuousPart
  val mPart:BDV[Double]=mask
  val label:Double=labelValue
  
  override def toString():String=
  {
    "C:"+cPart.toString()+"##M:"+mPart.toString()+"##L:"+label.toString()
  }
}