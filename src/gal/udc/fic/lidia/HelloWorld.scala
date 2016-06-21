package org.apache.spark.mllib.tests

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf

object HelloWorld
{
    def main(args: Array[String])
    {
      val conf = new SparkConf().setAppName("ADMCC-Test").setMaster("local")
      val sc=new SparkContext(conf)
      
    }
  }