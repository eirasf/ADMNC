# ADMNC
Anomaly detection in mixed numerical and categorical contexts

### Requirements:
  - (SBT)[https://www.scala-sbt.org/download.html]
  - (Spark 2.4)[https://spark.apache.org/downloads.html]

### Compilation:

    cd <PATH_TO_ADMNC>

    sbt clean assembly

### Execution:

    spark-submit [--master "local[NUM_THREADS]"] --class gal.udc.ADMNC <PATH_TO_JAR_FILE> <INPUT_DATASET> [options]

### Reference:
Eiras-Franco, C., Martinez-Rego, D., Guijarro-Berdinas, B., Alonso-Betanzos, A., & Bahamonde, A. (2019). Large scale anomaly detection in mixed numerical and categorical input spaces. Information Sciences, 487, 115-127.
