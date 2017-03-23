package gal.udc



object GibbsSampler extends Serializable
{
  var DEFAULT_BURN_IN:Int=1000
  
  /* Sequential version*/
  def obtainSampleWithFixedContinuousPart(sampleSize:Int, startingValue:MixedData, calculateFactor:(MixedData) => Double):Array[MixedData]=
    obtainSampleWithFixedContinuousPart(sampleSize:Int, startingValue:MixedData, calculateFactor:(MixedData) => Double, false)
  def obtainSampleWithFixedContinuousPart(sampleSize:Int, startingValue:MixedData, calculateFactor:(MixedData) => Double, ignoreLastVariable:Boolean):Array[MixedData]=
  {
    var result:Array[MixedData]=Array.ofDim[MixedData](sampleSize)
    var currentVar:Int=0
    val numVariables=startingValue.dPart.length
    var currentSample:MixedData=new MixedData(startingValue.cPart, startingValue.dPart.copy, startingValue.nValuesForDiscretes)
    for (i <- 0 until DEFAULT_BURN_IN)
    {
      currentSample.randomlySwitchDiscreteVariable(currentVar, calculateFactor)
      currentVar=(currentVar+1)%numVariables
    }
    for (i <- 0 until sampleSize)
    {
      currentSample.randomlySwitchDiscreteVariable(currentVar, calculateFactor)
      currentVar=(currentVar+1)%numVariables
      result(i)=new MixedData(startingValue.cPart, currentSample.dPart.copy, currentSample.nValuesForDiscretes)
    }
    return result
  }
}