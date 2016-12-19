package gal.udc

import scala.util.Random

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

import breeze.linalg.{ DenseVector => BDV }


object GenerateArtificialDataset
{
  val DEFAULT_ANOMALY_FRACTION:Double=0.01
  def showUsageAndExit()=
  {
    println("""Usage: GenerateArtificialDataset output_file_name [options]
    Saves a random dataset to a file with libsvm format
Options:
    -g num:mean:stddev\tGenerate num Gaussian samples with given mean and stdev. Can be repeated and there must be at least one.
    -f fraction\t\tFraction of labels that will be flipped (default: """"+DEFAULT_ANOMALY_FRACTION+""")""")
    System.exit(-1)
  }
  def parseParams(p:Array[String]):Map[String, Any]=
  {
    val m=scala.collection.mutable.Map[String, Any]("fraction" -> DEFAULT_ANOMALY_FRACTION)
    if (p.length<=0)
      showUsageAndExit()
    
    m("filename")=p(0)
    
    var gaussianParameters=List[GaussianSpec]()
    var continuousVariables:Int=0-1
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
          case "f"   => "fraction"
          case "g"   => "gaussian"
          case somethingElse => readOptionName
        }
      if (option=="gaussian")
      {
        val values=p(i+1).split(":")
        val num=values(0).toInt
        val mean=values(1).split("#").map({ x => x.toDouble })
        val stddev=values(2).split("#").map({ x => x.toDouble })
        gaussianParameters=gaussianParameters ::: List(new GaussianSpec(num, mean, stddev))
        
        if (continuousVariables<=0)
          continuousVariables=mean.length
          
        if ((continuousVariables!=mean.length) || (continuousVariables!=stddev.length))
        {
          println("Gaussians must have the same number of variables")
          showUsageAndExit()
        } 
      }
      else
      {
        if (!m.keySet.exists(_==option))
        {
          println("Unknown option:"+readOptionName)
          showUsageAndExit()
        }
        m(option)=p(i+1).toDouble
      }
      i=i+2
    }
    if (gaussianParameters.length<=0)
    {
      println("Parameters for at least one gaussian must be provided")
      showUsageAndExit()
    }
    m("gaussians")=gaussianParameters.toArray
    return m.toMap
  }
  def main(args: Array[String])
  {
    val options=parseParams(args)
    val sc=sparkContextSingleton.getInstance()
    
    //TEST - Generate synthetic data and test on it
    val listG=options("gaussians").asInstanceOf[Array[GaussianSpec]]
    val syntheticDataRDD=SyntheticDataGenerator.generate(sc, listG)
                          .map({ x => if (Random.nextFloat>options("fraction").asInstanceOf[Double]) x else x.makeAnomaly(listG.length)})
    
    MLUtils.saveAsLibSVMFile(syntheticDataRDD.map({ x => new LabeledPoint(if (x.isArtificialAnomaly) 1.0 else 0.0, new DenseVector(BDV.vertcat(x.dPart,x.cPart).toArray)) }),
                                                  options("filename").asInstanceOf[String])
  }
}
