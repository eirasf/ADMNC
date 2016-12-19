package gal.udc

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
import org.apache.spark.rdd.RDD

import breeze.linalg.{ DenseVector => BDV }
import breeze.linalg.{ SparseVector => BSV }
import breeze.linalg.{ Vector => BV }

class GaussianSpec(numElements:Long, meansG:Array[Double], sigmasG:Array[Double]) extends Serializable
{
  val nElems:Long=numElements
  val means:Array[Double]=meansG
  val sigmas:Array[Double]=sigmasG
}

object SyntheticDataGenerator
{
  //private def toBreeze(v: Vector): BV[Double] = v match
  private def toBreeze(v: Vector): BDV[Double] = v match
  {
    case DenseVector(values) => new BDV[Double](values)
    //case SparseVector(size, indices, values) => {new BSV[Double](indices, values, size)}
    case SparseVector(size, indices, values) => {new BDV[Double](v.toDense.values)}
  }
  private def toSpark(v: BV[Double]) = v match
  {
    case v: BDV[Double] => new DenseVector(v.toArray)
    case v: BSV[Double] => new SparseVector(v.length, v.index, v.data)
  }
  def generate(sc:SparkContext, continuousSpecs:Array[GaussianSpec]):RDD[MixedData]=
  {
    var totalRDD:RDD[MixedData]=null
    var i:Int=0
    
    for (i <- 0 until continuousSpecs.length)
    {
      val el=continuousSpecs(i) 
      val partialRDD=normalVectorRDD(sc, el.nElems, 1)
                      //.map(v => new MixedData(toBreeze(v)*el.sigmas+el.means,BDV(i,0.0,0.0,0.0)))
                      .map({v => val components=Array.fill(el.means.length)(v(0))
                                 for (j <- 0 until components.length)
                                   components(j)=components(j)*el.sigmas(j)+el.means(j)
                                 new MixedData(BDV(components),BDV(i))})
      if (totalRDD==null)
        totalRDD=partialRDD
      else
        totalRDD=totalRDD.union(partialRDD)
    }
    
    totalRDD
  }
}