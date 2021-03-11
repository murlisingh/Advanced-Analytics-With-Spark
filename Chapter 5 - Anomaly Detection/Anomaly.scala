import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession 
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql._
import org.apache.spark.broadcast._
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import scala.util.Random
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

object Anomaly extends App{
  
  Logger.getLogger("org").setLevel(Level.ERROR)
  
  val sparkConf = new SparkConf()
  sparkConf.set("spark.app.name","converting joining 2 tables")
  sparkConf.set("spark.master","local[*]")
  
  val spark = SparkSession.builder()
  .config(sparkConf)
  .getOrCreate()
  
  /* Loading the Data csv without headers*/
  val dataWithoutHeader = spark.read
  .option("inferSchema",true).option("header",false)
  .csv("C:/Users/mrajpuro/Downloads/kddcup.data.corrected")
  
  
  /* Reading the header and placing it*/
  val data = dataWithoutHeader.toDF(
 "duration", "protocol_type", "service", "flag",
 "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
 "hot", "num_failed_logins", "logged_in", "num_compromised",
 "root_shell", "su_attempted", "num_root", "num_file_creations",
 "num_shells", "num_access_files", "num_outbound_cmds",
 "is_host_login", "is_guest_login", "count", "srv_count",
 "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
 "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
 "dst_host_count", "dst_host_srv_count",
 "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
 "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
 "dst_host_serror_rate", "dst_host_srv_serror_rate",
 "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
 "label")
 
 data.show(5)
 
 /* Import necessary library to process Data Frame*/
 import spark.implicits._
 
 /* Explore data to count and see info present*/
 data.select("label").groupBy("label").count().orderBy($"count".desc).show(25)
 /*
 +----------------+-------+
|           label|  count|
+----------------+-------+
|          smurf.|2807886|
|        neptune.|1072017|
|         normal.| 972781|
|          satan.|  15892|
|        ipsweep.|  12481|
|      portsweep.|  10413|
|           nmap.|   2316|
|           back.|   2203|
|    warezclient.|   1020|
|       teardrop.|    979|
|            pod.|    264|
|   guess_passwd.|     53|
|buffer_overflow.|     30|
|           land.|     21|
|    warezmaster.|     20|
|           imap.|     12|
|        rootkit.|     10|
|     loadmodule.|      9|
|      ftp_write.|      8|
|       multihop.|      7|
|            phf.|      4|
|           perl.|      3|
|            spy.|      2|
+----------------+-------+*/
  
/* Dorpping all non-numeric columns for now*/ 
val numericOnly = data.drop("protocol_type", "service", "flag").cache()

/*A VectorAssembler creates a feature vector, a KMeans implementation creates a model from the feature vectors, and a Pipeline stitches it all
together. */
val assembler = new VectorAssembler().setInputCols(numericOnly.columns.filter(_ != "label")).setOutputCol("featureVector")

val kmeans = new KMeans().setPredictionCol("cluster").setFeaturesCol("featureVector")
  
val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
val pipelineModel = pipeline.fit(numericOnly)
val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
kmeansModel.clusterCenters.foreach(println)
/* [48.34019491959669,1834.6215497618625,826.2031900016945,5.7161172049003456E-6,6.487793027561892E-4,7.961734678254053E-6,0.012437658596734055,3.205108575604837E-5,0.14352904910348827,0.00808830584493399,6.818511237273984E-5,3.6746467745787934E-5,0.012934960793560386,0.0011887482315762398,7.430952366370449E-5,0.0010211435092468404,0.0,4.082940860643104E-7,8.351655530445469E-4,334.9735084506668,295.26714620807076,0.17797031701994256,0.1780369894027269,0.05766489875327379,0.05772990937912744,0.7898841322630906,0.02117961060991097,0.028260810096297884,232.98107822302248,189.21428335201279,0.7537133898007772,0.03071097882384052,0.605051930924901,0.006464107887636894,0.1780911843182284,0.1778858981346887,0.05792761150001272,0.05765922142401037]
[10999.0,0.0,1.309937401E9,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,255.0,1.0,0.0,0.65,1.0,0.0,0.0,0.0,1.0,1.0]*/

/* use the given labels to get an intuitive sense of what went into these two clusters by counting the labels within each cluster.*/
val withCluster = pipelineModel.transform(numericOnly)
withCluster.select("cluster", "label").
 groupBy("cluster", "label").count().
 orderBy($"cluster", $"count".desc).
 show(25)
 
 /*
 +-------+----------------+-------+
|cluster|           label|  count|
+-------+----------------+-------+
|      0|          smurf.|2807886|
|      0|        neptune.|1072017|
|      0|         normal.| 972781|
|      0|          satan.|  15892|
|      0|        ipsweep.|  12481|
|      0|      portsweep.|  10412|
|      0|           nmap.|   2316|
|      0|           back.|   2203|
|      0|    warezclient.|   1020|
|      0|       teardrop.|    979|
|      0|            pod.|    264|
|      0|   guess_passwd.|     53|
|      0|buffer_overflow.|     30|
|      0|           land.|     21|
|      0|    warezmaster.|     20|
|      0|           imap.|     12|
|      0|        rootkit.|     10|
|      0|     loadmodule.|      9|
|      0|      ftp_write.|      8|
|      0|       multihop.|      7|
|      0|            phf.|      4|
|      0|           perl.|      3|
|      0|            spy.|      2|
|      1|      portsweep.|      1|
+-------+----------------+-------+  The result shows that the clustering was not at all helpful. Only one data point ended
up in cluster 1!*/
 
/* KMeansModel offers a computeCost method that computes the sum of squared distances and can easily be used to compute the mean squared distance.*/
/* Try 1 --> def clusteringScore0(data: DataFrame, k: Int): Double = {
 val assembler = new VectorAssembler().
 setInputCols(data.columns.filter(_ != "label")).setOutputCol("featureVector")
 
 val kmeans = new KMeans().setSeed(Random.nextLong()).setK(k).setPredictionCol("cluster").setFeaturesCol("featureVector")
 val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
 val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
 kmeansModel.computeCost(assembler.transform(data)) / data.count() /*Compute mean from total squared distance (“cost”)*/
}
(20 to 100 by 20).map(k => (k, clusteringScore0(numericOnly, k))).foreach(println)
/* (20,1.2884868962389682E8)
(40,1.7235903703754384E7)
(60,1.5122264124864498E7)
(80,1.439888123260484E7)
(100,1.5974615151368191E7)* Values decreasing as K is increasing  <-- */

/* Try 2 --> We can improve it by running the iteration longer. The algorithm has a threshold via
setTol() that controls the minimum amount of cluster centroid movement consid‐
ered significant; lower values mean the K-means algorithm will let the centroids con‐
tinue to move longer. Increasing the maximum number of iterations with
setMaxIter() also prevents it from potentially stopping too early at the cost of possi‐
bly more computation.*/
def clusteringScore1(data: DataFrame, k: Int): Double = {
 val assembler = new VectorAssembler().
 setInputCols(data.columns.filter(_ != "label")).setOutputCol("featureVector")
 
 val kmeans = new KMeans().setSeed(Random.nextLong()).setK(k).setPredictionCol("cluster").setFeaturesCol("featureVector").setMaxIter(40).setTol(1.0e-5)
 val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
 val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
 kmeansModel.computeCost(assembler.transform(data)) / data.count() /*Compute mean from total squared distance (“cost”)*/
}
(20 to 100 by 20).map(k => (k, clusteringScore1(numericOnly, k))).foreach(println)
(20,1.0730295992159383E8)
(40,1.214689594269054E8)
(60,1.1590273751929056E7)
(80,8620789.19669756)
(100,9501486.12438394) */

/* Feature Normalization */
def clusteringScore2(data: DataFrame, k: Int): Double = {
 val assembler = new VectorAssembler().
 setInputCols(data.columns.filter(_ != "label")).
 setOutputCol("featureVector")
 val scaler = new StandardScaler()
 .setInputCol("featureVector")
 .setOutputCol("scaledFeatureVector")
 .setWithStd(true)
 .setWithMean(false)
 val kmeans = new KMeans().
 setSeed(Random.nextLong()).
 setK(k).
 setPredictionCol("cluster").
 setFeaturesCol("scaledFeatureVector").
 setMaxIter(40).
 setTol(1.0e-5)
 val pipeline = new Pipeline().setStages(Array(assembler, scaler, kmeans))
 val pipelineModel = pipeline.fit(data)
 val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
 kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
}
(60 to 270 by 30).map(k => (k, clusteringScore2(numericOnly, k))).
 foreach(println)
 
 
 /* First, the string values are converted to integer indices like 0, 1, 2, and so on
using StringIndexer. Then these integer indices are encoded into a vector with One
HotEncoder.*/
 
 def oneHotPipeline(inputCol: String): (Pipeline, String) = {
 val indexer = new StringIndexer().
 setInputCol(inputCol).
 setOutputCol(inputCol + "_indexed")
 val encoder = new OneHotEncoder().
 setInputCol(inputCol + "_indexed").
 setOutputCol(inputCol + "_vec")
 val pipeline = new Pipeline().setStages(Array(indexer, encoder))
 (pipeline, inputCol + "_vec")
}

  def clusteringScore3(data: DataFrame, k: Int): Double = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
    val pipelineModel = pipeline.fit(data)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.summary.trainingCost
  }

  def clusteringTake3(data: DataFrame): Unit = {
    (60 to 270 by 30).map(k => (k, clusteringScore3(data, k))).foreach(println)
  }

  /* Entropy*/ 
 def entropy(counts: Iterable[Int]): Double = {
 val values = counts.filter(_ > 0)
 val n = values.map(_.toDouble).sum
 values.map { v =>
 val p = v / n
 -p * math.log(p)
 }.sum
}
 
   def fitPipeline4(data: DataFrame, k: Int): PipelineModel = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
    pipeline.fit(data)
  }
 
   def clusteringScore4(data: DataFrame, k: Int): Double = {
    val pipelineModel = fitPipeline4(data, k)

    // Predict cluster for each datum
    val clusterLabel = pipelineModel.transform(data).
      select("cluster", "label").as[(Int, String)]
    val weightedClusterEntropy = clusterLabel.
      // Extract collections of labels, per cluster
      groupByKey { case (cluster, _) => cluster }.
      mapGroups { case (_, clusterLabels) =>
        val labels = clusterLabels.map { case (_, label) => label }.toSeq
        // Count labels in collections
        val labelCounts = labels.groupBy(identity).values.map(_.size)
        labels.size * entropy(labelCounts)
      }.collect()

    // Average entropy weighted by cluster size
    weightedClusterEntropy.sum / data.count()
  }

  def clusteringTake4(data: DataFrame): Unit = {
    (60 to 270 by 30).map(k => (k, clusteringScore4(data, k))).foreach(println)

    val pipelineModel = fitPipeline4(data, 180)
    val countByClusterLabel = pipelineModel.transform(data).
      select("cluster", "label").
      groupBy("cluster", "label").count().
      orderBy("cluster", "label")
    countByClusterLabel.show()
  }
  
  // Anomaly Detector
  def buildAnomalyDetector(data: DataFrame): Unit = {
    val pipelineModel = fitPipeline4(data, 180)

    val kMeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    val centroids = kMeansModel.clusterCenters

    val clustered = pipelineModel.transform(data)
    val threshold = clustered.
      select("cluster", "scaledFeatureVector").as[(Int, Vector)].
      map { case (cluster, vec) => Vectors.sqdist(centroids(cluster), vec) }.
      orderBy($"value".desc).take(100).last

    val originalCols = data.columns
    val anomalies = clustered.filter { row =>
      val cluster = row.getAs[Int]("cluster")
      val vec = row.getAs[Vector]("scaledFeatureVector")
      Vectors.sqdist(centroids(cluster), vec) >= threshold
    }.select(originalCols.head, originalCols.tail:_*)

    println(anomalies.first())
  }

}





 

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
}