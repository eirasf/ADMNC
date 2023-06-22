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
import java.io.PrintWriter

class MultivariateGaussianSpec(numElements:Long, meansG:Array[Double], sigmasG:Array[Double]) extends Serializable
{
  val nElems:Long=numElements
  val means:Array[Double]=meansG
  val sigmas:Array[Double]=sigmasG
}

class GaussianSpec(numElements:Long, meanG:Double, sigmaG:Double) extends Serializable
{
  val nElems:Long=numElements
  val mean:Double=meanG
  val sigma:Double=sigmaG
}

object SyntheticDataGenerator
{
  private val PARALLELISM=100//10000
  private val DEFAULT_BINARY_PROBABILITY=0.5
  val DEFAULT_NUM_TRAIN=2500
  val DEFAULT_NUM_DEV=500
  val DEFAULT_NUM_TEST=500
  val DEFAULT_NUM_RANDOM_CATEGORICAL=20
  val DEFAULT_NUM_RELATED_CATEGORICAL=10
  val DEFAULT_NUM_CONTINUOUS=100
  val DEFAULT_NOISE_SIGMA=1.0
  val DEFAULT_NOISE_VARIABLES=32
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
  /**
 * @param sc
 * @param continuousSpecs
 * @return Generates a dataset with Gaussians identified with labels. Anomalies consist of a flipped label.
 */
def generate(sc:SparkContext, continuousSpecs:Array[MultivariateGaussianSpec]):RDD[MixedData]=
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
  
  def generateWithRules(sc:SparkContext, numElems:Int, numCategorical:Int, rules:Array[(Seq[Int],Seq[GaussianSpec])], defaultDist:Seq[GaussianSpec], paretoXm:Int=2, paretoAlpha:Double=1.0, pw:PrintWriter):RDD[MixedData]=
  {
    val seedArray=new Array[Int](100)
    val startRDD=sc.parallelize(seedArray, 100)
    val elemsPerPartition=numElems/100.0
    val bRules=sc.broadcast(rules)
    
    val vectorsRDD=startRDD.flatMap { x => var n=0
                            val elems=new Array[(BDV[Double],BDV[Double], Seq[Int])](elemsPerPartition.ceil.toInt)
                            for (n <- 0 until elems.length)
                            {
                              var vector=new Array[Double](numCategorical)
                              var i=0
                              for (i <- 0 until vector.length)
                                if (Math.random()<Math.pow(paretoXm/(paretoXm+i+1.0),paretoAlpha))
                                  vector(i)=1
                              val lRules=bRules.value
                              var cVector:BDV[Double]=null //Initialize with default distribution?
                              var ruleNumbers = List[Int]()
                              i=0
                              while (i < lRules.length)
                              {
                                val r=lRules(i)
                                var verifies=true
                                var index=0
                                while (verifies && (index < r._1.length))
                                {
                                  verifies=vector(index)==1.0
                                  index+=1
                                }
                                if (verifies)
                                {
                                  val newVector=generateContinuousVector(r._2)
                                  if (cVector==null)
                                    cVector=newVector
                                  else
                                    cVector=cVector+newVector
                                  ruleNumbers=ruleNumbers :+ i
                                }
                                i+=1
                              }
                              if (cVector==null)
                                cVector=generateContinuousVector(defaultDist)
                              if (ruleNumbers.length==0)
                                ruleNumbers=ruleNumbers :+ -1
                              elems(n)=(BDV(vector),cVector,ruleNumbers)
                            }
                            elems
                      }
            
    
    //vectorsRDD.take(10).foreach { x => println(x._1.toArray.mkString("|")+" # "+x._2.toArray.mkString("|")+" from rule(s) "+x._3.mkString("|")) }
    val totalElements=vectorsRDD.count()
    vectorsRDD.flatMap({ x => x._3.map { y => (y,1) } }).reduceByKey(_+_).collect().sortBy(-_._2).foreach({x => println((x._2/numElems.toDouble).formatted("%.4f")+" elements from rule "+x._1)
                                                                                                                pw.println((x._2/numElems.toDouble).formatted("%.4f")+" elements from rule "+x._1)})
    return vectorsRDD.map({x => new MixedData(x._2, x._1)})
  }
  
  def generateContinuousVector(gaussianSpecs:Seq[GaussianSpec]):BDV[Double]=
  {
    val gRandom=new scala.util.Random(System.nanoTime())
    val vector=new Array[Double](gaussianSpecs.length)
    var i=0
    for (i <- 0 until vector.length)
      vector(i)=gRandom.nextDouble()*gaussianSpecs(i).sigma+gaussianSpecs(i).mean
    return BDV(vector)
  }
  
  /**
 * @param sc
 * @param numElems
 * @param numRandomCategorical
 * @param numRelatedCategorical
 * @param numNumerical
 * @param beta
 * @param paretoXm
 * @param paretoAlpha
 * @return Generates a dataset without anomalies
 */
def generate(sc:SparkContext, numElems:Int, numRandomCategorical:Int, numRelatedCategorical:Int, numNumerical:Int, binaryProbability:Double=DEFAULT_BINARY_PROBABILITY):RDD[MixedData]=
  {
    var totalRDD:RDD[MixedData]=null
    
    val startRDD=sc.parallelize(Array.fill(PARALLELISM)(0), PARALLELISM)
    val elemsPerPartition=Math.ceil(numElems.toDouble/PARALLELISM).toInt
    
    val randomBinaryPart=startRDD.flatMap(
              { x =>
                var vectors=new Array[BDV[Double]](elemsPerPartition)
                var i=0
                for (i <- 0 until vectors.length)
                {
                  var vector=new Array[Double](numRandomCategorical)
                  var j=0
                  for (j <- 0 until vector.length)
                    if (Math.random()<binaryProbability)
                      vector(j)=1
                  vectors(i)=BDV(vector)
                }
                vectors
              })
              
    var generatedRBPart=randomBinaryPart
    
    //Ensure that we don't return many more elements than asked.
    var numElemsGenerated=generatedRBPart.count()
    var attempts=0
    //while (((numElemsGenerated>numElems+10) || (numElemsGenerated<numElems)) && (attempts<10))
    while ((numElemsGenerated<numElems) && (attempts<10))
    {
      generatedRBPart=randomBinaryPart.sample(false, numElems.toDouble/randomBinaryPart.count())
      numElemsGenerated=generatedRBPart.count()
    }
    
    val binaryRules=_generateRules(numRelatedCategorical, numRandomCategorical)
    val bBinaryRules=sc.broadcast(binaryRules)
    val generatedBinaryPart=generatedRBPart.map(
          { x =>
            val bRules=bBinaryRules.value
            val values=bRules.map { r =>
              val p=x.dot(r)/breeze.linalg.sum(r)
              if (Math.random()<p)
                1.0
              else
                0.0
              }
            //BDV.vertcat(x, BDV(values))
            BDV.vertcat(BDV(values), x)
          })
          
    val numericalRules=_generateRules(numNumerical, numRelatedCategorical+numRandomCategorical)
    val bNumericalRules=sc.broadcast(numericalRules)
    val generatedDataset=generatedBinaryPart.map(
          { x =>
            val bRules=bNumericalRules.value
            var generatedElement:MixedData=null
            
            val values=bRules.map { r =>
                                        val p=x.dot(r)/breeze.linalg.sum(r)
                                        new GaussianSpec(1, p, 1/(p+1))
                                        }
            new MixedData(generateContinuousVector(values), x)
          })
    
    generatedDataset
  }
  
  private def _generateRules(numRules:Int, ruleSize:Int):Array[BDV[Double]]=
  {
    val ruleComponentProbability=1.0/ruleSize
    val binaryRules=new Array[BDV[Double]](numRules)
    var i=0
    for (i <- 0 until binaryRules.length)
    {
      var vector=new Array[Double](ruleSize)
      var j=0
      var numOnes=0
      while(numOnes<2)
      {
        for (j <- 0 until vector.length)
          if (Math.random()<ruleComponentProbability)
            vector(j)=1
        binaryRules(i)=BDV(vector)
        numOnes=vector.sum.toInt
      }
    }
    binaryRules
  }
}