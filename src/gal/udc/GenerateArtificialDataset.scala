package gal.udc

import scala.util.Random

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

import breeze.linalg.{ DenseVector => BDV }
import java.io._


object GenerateArtificialDataset
{
  val DEFAULT_ANOMALY_FRACTION:Double=0.01
  def showUsageAndExit()=
  {
    println(s"""Usage: GenerateArtificialDataset output_file_name [options]
    Saves a random dataset to a file with libsvm format
Options:
    -f fraction         Fraction of labels that will be flipped (default: ${DEFAULT_ANOMALY_FRACTION})
    -ntr numTrain    Number of elements in the training dataset (default: ${SyntheticDataGenerator.DEFAULT_NUM_TRAIN})
    -nd numDev          Number of elements in the dev dataset (default: ${SyntheticDataGenerator.DEFAULT_NUM_DEV})
    -nts numTest        Number of elements in the test dataset (default: ${SyntheticDataGenerator.DEFAULT_NUM_TEST})
    -rc numRandomCat    Number of random categorical variables (default: ${SyntheticDataGenerator.DEFAULT_NUM_RANDOM_CATEGORICAL})
    -dc numDependentCat Number of dependent categorical variables (default: ${SyntheticDataGenerator.DEFAULT_NUM_RELATED_CATEGORICAL})
    -nc numContinuous   Number of continuous variables (default: ${SyntheticDataGenerator.DEFAULT_NUM_CONTINUOUS})
    -nv noiseVariables  Number of variables affected by noise (default: ${SyntheticDataGenerator.DEFAULT_NOISE_VARIABLES})
    -nm noiseMagnitude  Magnitude of the gaussian noise added to anomalies (default: ${SyntheticDataGenerator.DEFAULT_NOISE_SIGMA})""")
    System.exit(-1)
  }
  def parseParams(p:Array[String]):Map[String, Any]=
  {
    val m=scala.collection.mutable.Map[String, Any]("fraction" -> DEFAULT_ANOMALY_FRACTION,
                                                    "numTrain"-> SyntheticDataGenerator.DEFAULT_NUM_TRAIN.toDouble,
                                                    "numDev"-> SyntheticDataGenerator.DEFAULT_NUM_DEV.toDouble,
                                                    "numTest"-> SyntheticDataGenerator.DEFAULT_NUM_TEST.toDouble,
                                                    "numRandomCat"-> SyntheticDataGenerator.DEFAULT_NUM_RANDOM_CATEGORICAL.toDouble,
                                                    "numDependentCat"-> SyntheticDataGenerator.DEFAULT_NUM_RELATED_CATEGORICAL.toDouble,
                                                    "numContinuous"-> SyntheticDataGenerator.DEFAULT_NUM_CONTINUOUS.toDouble,
                                                    "noiseVariables"-> SyntheticDataGenerator.DEFAULT_NOISE_VARIABLES.toDouble,
                                                    "noiseMagnitude"-> SyntheticDataGenerator.DEFAULT_NOISE_SIGMA)
    if (p.length<=0)
      showUsageAndExit()
    
    m("filename")=p(0)
    
    var gaussianParameters=List[MultivariateGaussianSpec]()
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
          case "ntr"   => "numTrain"
          case "nd"   => "numDev"
          case "nts"   => "numTest"
          case "rc"   => "numRandomCat"
          case "dc"   => "numDependentCat"
          case "nc"   => "numContinuous"
          case "nv"   => "noiseVariables"
          case "nm"   => "noiseMagnitude"
          case somethingElse => readOptionName
        }

      if (!m.keySet.exists(_==option))
      {
        println("Unknown option:"+readOptionName)
        showUsageAndExit()
      }
      m(option)=p(i+1).toDouble
      i=i+2
    }
    /*if (gaussianParameters.length<=0)
    {
      println("Parameters for at least one gaussian must be provided")
      showUsageAndExit()
    }
    m("gaussians")=gaussianParameters.toArray*/
    return m.toMap
  }
  def main(args: Array[String])
  {
    val options=parseParams(args)
    val sc=sparkContextSingleton.getInstance()
    
    val NUM_TRAIN=options("numTrain").asInstanceOf[Double].toInt
    val NUM_DEV=options("numDev").asInstanceOf[Double].toInt
    val NUM_TEST=options("numTest").asInstanceOf[Double].toInt
    val NUM_RANDOM_CATEGORICAL=options("numRandomCat").asInstanceOf[Double].toInt
    val NUM_RELATED_CATEGORICAL=options("numDependentCat").asInstanceOf[Double].toInt
    val NUM_CONTINUOUS=options("numContinuous").asInstanceOf[Double].toInt
    val NOISE_SIGMA=options("noiseMagnitude").asInstanceOf[Double]
    val NOISE_VARIABLES=options("noiseVariables").asInstanceOf[Double].toInt
    val ANOMALY_FRACTION=options("fraction").asInstanceOf[Double]
    val filename=options("filename").asInstanceOf[String]
    val pw = new PrintWriter(new File(filename.replace(".libsvm_", "-info.txt")))
    
    val generatedRDD=SyntheticDataGenerator.generate(sc, NUM_TRAIN+NUM_DEV+NUM_TEST, NUM_RANDOM_CATEGORICAL, NUM_RELATED_CATEGORICAL, NUM_CONTINUOUS)
    
    val generatedArray=generatedRDD.collect()
    
    if (NUM_TRAIN>0)
    {
      val trainRDD=sc.parallelize(generatedArray.slice(0, NUM_TRAIN), 4)
      MLUtils.saveAsLibSVMFile(trainRDD.map({ x => new LabeledPoint(if (x.isArtificialAnomaly) 1.0 else 0.0, new DenseVector(BDV.vertcat(x.dPart,x.nPart).toArray)) }),
                                                    filename.replace(".libsvm", "_train.libsvm"))
    }
    val devRDDNonAnomalous=sc.parallelize(generatedArray.slice(NUM_TRAIN, NUM_TRAIN+NUM_DEV), 4)
    val devRDD=devRDDNonAnomalous.map { x => if (Math.random()<ANOMALY_FRACTION)
                                                x.makeAnomaly(NOISE_SIGMA, NOISE_VARIABLES)
                                              else
                                                x
                                              }
    
    if (NUM_DEV>0)
    {
      //DEBUG
      //MLUtils.saveAsLibSVMFile(devRDDNonAnomalous.map({ x => new LabeledPoint(if (x.isArtificialAnomaly) 1.0 else 0.0, new DenseVector(BDV.vertcat(x.dPart,x.cPart).toArray)) }),
      //                                              filename.replace(".libsvm", "_dev-preanomalies.libsvm"))
      val devFileName=if (NUM_TEST+NUM_TRAIN>0) filename.replace(".libsvm", "_dev.libsvm") else filename
      MLUtils.saveAsLibSVMFile(devRDD.map({ x => new LabeledPoint(if (x.isArtificialAnomaly) 1.0 else 0.0, new DenseVector(BDV.vertcat(x.dPart,x.nPart).toArray)) }),
                                                    devFileName)
    }

    if (NUM_TEST>0)
    {
      val testRDDNonAnomalous=sc.parallelize(generatedArray.slice(NUM_TRAIN+NUM_DEV, NUM_TRAIN+NUM_DEV+NUM_TEST), 4)
      val testRDD=testRDDNonAnomalous.map { x => if (Math.random()<ANOMALY_FRACTION)
                                                  x.makeAnomaly(NOISE_SIGMA, NOISE_VARIABLES)
                                                else
                                                  x
                                                }
      MLUtils.saveAsLibSVMFile(testRDD.map({ x => new LabeledPoint(if (x.isArtificialAnomaly) 1.0 else 0.0, new DenseVector(BDV.vertcat(x.dPart,x.nPart).toArray)) }),
                                                    filename.replace(".libsvm", "_test.libsvm"))
    }
    
    val summaryMessage=NUM_TRAIN+" training samples | "+NUM_DEV+" dev samples | "+NUM_TEST+" test samples\n\t"+NUM_RANDOM_CATEGORICAL+" random categoricals\n\t"+NUM_RELATED_CATEGORICAL+" related categoricals \n\t"+NUM_CONTINUOUS+" numerical\n\tAnomaly fraction="+ANOMALY_FRACTION+"\n\tNoise sigma="+NOISE_SIGMA+"\n\tNoise variables="+NOISE_VARIABLES
    println("File "+filename+" saved")
    pw.println("File "+filename+" saved")
    println(summaryMessage)
    pw.println(summaryMessage)
    pw.close
  }
}