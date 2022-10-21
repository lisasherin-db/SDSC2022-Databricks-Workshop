# Databricks notebook source
import random 
from pyspark.sql.functions import monotonically_increasing_id

def random_location():
  lat = random.uniform(43.1426780419647, 41.730950572874264)
  lon = random.uniform(-72.77125552296638,-71.01084798574448)
  return {"longitude": lon, "latitude": lat}

spark.sql("CREATE DATABASE IF NOT EXISTS isochrones")
locations = [random_location() for i in range(50)]
points_of_origin = spark.createDataFrame(locations).withColumn("id", monotonically_increasing_id())
points_of_origin.write.format("delta").mode("overwrite").saveAsTable(f"isochrones.potential_locations")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT * FROM isochrones.potential_locations