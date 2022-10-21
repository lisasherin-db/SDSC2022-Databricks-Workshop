# Databricks notebook source
# MAGIC %md ## Load DIY Stores and Furniture Stores
# MAGIC 
# MAGIC Getting furniture stores OR DIY stores directly from OSM using the [overpass API.](https://overpass-turbo.eu/s/1mKU)

# COMMAND ----------

# MAGIC %pip install -q geopandas databricks-mosaic

# COMMAND ----------

import geopandas as gpd
import mosaic as mos
from pyspark.sql.functions import *

mos.enable_mosaic(spark, dbutils)

gpd_stores = gpd.read_file("file:/Workspace/Repos/timo.roest@databricks.com/SDSC2022-workshop/Isochrones/Data Prep/stores.geojson")
gpd_stores['wkt'] = gpd_stores.geometry.to_wkt()
gpd_stores = gpd_stores[["id", "name", "wkt"]]
stores = (
  spark.createDataFrame(gpd_stores)
  .withColumn("geom", mos.st_geomfromwkt("wkt"))
)

(stores
.write
.format("delta")
.mode("overwrite")
.option("overwriteSchema", "true")
.saveAsTable("isochrones.stores")
)

# COMMAND ----------

display(stores)

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC stores geom geometry

# COMMAND ----------


