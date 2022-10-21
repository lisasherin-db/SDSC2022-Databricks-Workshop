# Databricks notebook source
# MAGIC %pip install -q databricks-mosaic

# COMMAND ----------

# MAGIC %md ## Loading Isochrones

# COMMAND ----------

import mosaic as mos
from pyspark.sql.functions import *

spark.conf.set("spark.databricks.labs.mosaic.geometry.api", "JTS")
mos.enable_mosaic(spark, dbutils)

# COMMAND ----------

isochrones = spark.read.table("isochrones.isochrones")

# COMMAND ----------

neighbourhoods_mosaic_frame = mos.MosaicFrame(isochrones, "geom")
optimal_resolution = 8 #neighbourhoods_mosaic_frame.get_optimal_resolution(sample_fraction=1.0)

# COMMAND ----------

chipped_isochrones = (
  isochrones
  .withColumn("mosaic", mos.mosaic_explode("geom", lit(optimal_resolution)))
  .select(
    "location_id", "time_min", col("mosaic.wkb").alias("chipped_geom"), "mosaic", col("mosaic.index_id").alias("h3")
  )
)

chipped_isochrones.createOrReplaceTempView("chipped_isochrones")
display(chipped_isochrones)

# COMMAND ----------

sampled_isochrones = chipped_isochrones.filter("location_id IN (841813590016)").orderBy(desc("time_min"))

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC sampled_isochrones chipped_geom geometry 30_000

# COMMAND ----------

# MAGIC %md ## Leverage Reference Data

# COMMAND ----------

# MAGIC %md ### Population

# COMMAND ----------

resolution = lit(8)
global_population = spark.sql("SELECT h3_stringtoh3(h3) AS h3, population FROM isochrones.global_population")
display(global_population)

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH population AS (
# MAGIC   SELECT 
# MAGIC     h3_stringtoh3(h3) as h3, 
# MAGIC     population 
# MAGIC   FROM 
# MAGIC     isochrones.global_population
# MAGIC )
# MAGIC SELECT
# MAGIC   location_id, 
# MAGIC   time_min,
# MAGIC   sum(population) as population
# MAGIC FROM 
# MAGIC   chipped_isochrones AS iso
# MAGIC LEFT JOIN 
# MAGIC   population AS pop
# MAGIC ON 
# MAGIC   pop.h3 = iso.h3
# MAGIC GROUP BY 
# MAGIC   location_id,
# MAGIC   time_min
# MAGIC ORDER BY 
# MAGIC   location_id DESC,
# MAGIC   time_min ASC

# COMMAND ----------

# MAGIC %md ### Stores

# COMMAND ----------

stores = (
  spark
  .read
  .table("isochrones.stores")
  .withColumn("centroid", mos.st_centroid2D("geom"))
  .withColumn("h3", mos.point_index_lonlat("centroid.x", "centroid.y", resolution))
)
stores.cache()
stores.createOrReplaceTempView("stores")

# COMMAND ----------

display(stores)

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC stores h3 h3

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   iso.location_id, 
# MAGIC   iso.time_min,
# MAGIC   count(sto.id) as no_stores
# MAGIC FROM 
# MAGIC   chipped_isochrones AS iso
# MAGIC LEFT JOIN 
# MAGIC   stores AS sto
# MAGIC ON 
# MAGIC   sto.h3 = iso.mosaic.index_id
# MAGIC GROUP BY 
# MAGIC   location_id,
# MAGIC   time_min
# MAGIC ORDER BY 
# MAGIC   location_id DESC,
# MAGIC   time_min ASC

# COMMAND ----------

# MAGIC %md ### Combined Analysis
# MAGIC 
# MAGIC We can now combine the two data sources to inform us to give us a singular view of the data. 

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW store_location_analysis AS
# MAGIC WITH population AS (
# MAGIC   SELECT 
# MAGIC     h3_stringtoh3(h3) as h3, 
# MAGIC     population 
# MAGIC   FROM 
# MAGIC     isochrones.global_population
# MAGIC )
# MAGIC SELECT
# MAGIC   location_id, 
# MAGIC   time_min,
# MAGIC   sum(population) as population,
# MAGIC   count(sto.id) as no_stores
# MAGIC FROM 
# MAGIC   chipped_isochrones AS iso
# MAGIC LEFT JOIN 
# MAGIC   population AS pop
# MAGIC ON 
# MAGIC   pop.h3 = iso.h3
# MAGIC LEFT JOIN 
# MAGIC   stores AS sto
# MAGIC ON 
# MAGIC   sto.h3 = iso.mosaic.index_id
# MAGIC GROUP BY 
# MAGIC   location_id,
# MAGIC   time_min
# MAGIC ORDER BY 
# MAGIC   location_id DESC,
# MAGIC   time_min ASC;
# MAGIC SELECT * FROM store_location_analysis;

# COMMAND ----------

# MAGIC %md ## Conclusion
# MAGIC 
# MAGIC Since our ultimate goal was to identify the top new location that will serve the most customers and stores within a 60 minutes, we can now easily query this data for that result. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC location_id, 
# MAGIC population,
# MAGIC no_stores
# MAGIC FROM store_location_analysis
# MAGIC WHERE time_min = 60
# MAGIC ORDER BY population DESC, no_stores DESC
# MAGIC LIMIT 10

# COMMAND ----------

top_candidate = chipped_isochrones.filter("location_id IN (171798691840)").orderBy(desc("time_min"))

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC top_candidate chipped_geom geometry 30_000

# COMMAND ----------


