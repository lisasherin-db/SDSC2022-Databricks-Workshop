# Databricks notebook source
# MAGIC %md #Download Data

# COMMAND ----------

# --- UNCOMMENT TO TEST IF EXISTS ---
# dbutils.fs.ls("/datasets/graphhopper/osm/us-northeast")

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/datasets/graphhopper/osm/us-northeast/")
dbutils.fs.mkdirs("dbfs:/datasets/graphhopper/osm/us-northeast/graphHopperData")

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /dbfs/datasets/graphhopper/osm/us-northeast/
# MAGIC wget -q https://download.geofabrik.de/north-america/us-northeast-latest.osm.pbf

# COMMAND ----------

dbutils.fs.ls("/datasets/graphhopper/osm/us-northeast/")

# COMMAND ----------


