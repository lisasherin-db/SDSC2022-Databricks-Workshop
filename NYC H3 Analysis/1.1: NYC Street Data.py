# Databricks notebook source
# MAGIC %md # NYC Street Data
# MAGIC 
# MAGIC > CSCL PUB Centerline is a road-bed representation of New York City streets containing address ranges and other information such as traffic directions, road types, segment types [[1](https://data.cityofnewyork.us/City-Government/road/svwp-sbcd)]
# MAGIC 
# MAGIC --- 
# MAGIC __Author: Michael Johns <mjohns@databricks.com>__  
# MAGIC _Last Modified: 17 OCT 2022_

# COMMAND ----------

# MAGIC %md ## Setup
# MAGIC <p/>
# MAGIC 
# MAGIC 1. Import Databricks columnar functions (including H3) for DBR / DBSQL Photon with `from pyspark.databricks.sql.functions import *`
# MAGIC 1. To use Databricks Labs [Mosaic](https://databrickslabs.github.io/mosaic/index.html) library for geospatial data engineering, analysis, and visualization functionality:
# MAGIC   * Install with `%pip install databricks-mosaic`
# MAGIC   * Import and use with the following:
# MAGIC   ```
# MAGIC   import mosaic as mos
# MAGIC   mos.enable_mosaic(spark, dbutils)
# MAGIC 
# MAGIC   ```
# MAGIC <p/>
# MAGIC 
# MAGIC 3. To use [KeplerGl](https://kepler.gl/) OSS library for map layer rendering:
# MAGIC   * Already installed with Mosaic, use `%%mosaic_kepler` magic [[Mosaic Docs](https://databrickslabs.github.io/mosaic/usage/kepler.html)]
# MAGIC   * Import with `from keplergl import KeplerGl` to use directly
# MAGIC 
# MAGIC __Note: If you hit `H3_NOT_ENABLED` [[docs](https://docs.databricks.com/error-messages/h3-not-enabled-error-class.html#h3_not_enabled-error-class)]__
# MAGIC 
# MAGIC > `h3Expression` is disabled or unsupported. Consider enabling Photon or switch to a tier that supports H3 expressions. [[AWS](https://www.databricks.com/product/aws-pricing) | [Azure](https://azure.microsoft.com/en-us/pricing/details/databricks/) | [GCP](https://www.databricks.com/product/gcp-pricing)]
# MAGIC 
# MAGIC __Recommend running on DBR 11.3+ for maximizing photonized functions.__

# COMMAND ----------

# MAGIC %pip install databricks-mosaic --quiet

# COMMAND ----------

from pyspark.databricks.sql.functions import *

from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.sql import Window

import mosaic as mos
mos.enable_mosaic(spark, dbutils)

# COMMAND ----------

# MAGIC %run ./Helpers/nyc_data_config.py

# COMMAND ----------

# MAGIC %run ./Helpers/nyc_table_helper.py

# COMMAND ----------

# MAGIC %md ### Download GeoJSON (1x)

# COMMAND ----------

#%sh 
# wget -O nyc_streets.geojson "https://data.cityofnewyork.us/api/geospatial/svwp-sbcd?method=export&format=GeoJSON"
# mkdir -p $etl_dir_fuse
# cp nyc_streets.geojson $etl_dir_fuse/

# COMMAND ----------

display(dbutils.fs.ls(etl_dir))

# COMMAND ----------

# MAGIC %md __Check Data__
# MAGIC > Just initial explode on `features`, will refine further down below

# COMMAND ----------

# df_test = (
#    spark.read
#     .option("multiline", "true")
#     .format("json")
#     .load(f"{etl_dir}/nyc_streets.geojson")
#       .select("type", F.explode(col("features")).alias("feature"))
#       .select("type", col("feature.properties").alias("properties"), F.to_json(col("feature.geometry")).alias("json_geometry"))
#     .limit(3)
# )
# display(df_test)

# COMMAND ----------

# MAGIC %md ## Load Streets
# MAGIC 
# MAGIC > Will use built-in JSON with spark and raise properties to table level; also, helpful to reference the [Streets Data Dictionary](https://gis.ny.gov/gisdata/supportfiles/Streets-Data-Dictionary.pdf) and [NYC Planning](https://nycplanning.github.io/Geosupport-UPG/chapters/chapterIV/section09/)
# MAGIC 
# MAGIC __Mosaic Functions__
# MAGIC 
# MAGIC ```
# MAGIC .withColumn("geom", mos.st_geomfromgeojson("json_geometry"))
# MAGIC .withColumn("geom", mos.st_setsrid("geom", F.lit(4326)))
# MAGIC .withColumn("geom_wkt", mos.st_astext("geom"))
# MAGIC .withColumn("is_valid", mos.st_isvalid("geom"))
# MAGIC .withColumn("is_coords_4326", mos.st_hasvalidcoordinates("geom", F.lit("EPSG:4326"), F.lit('bounds')))
# MAGIC ```

# COMMAND ----------

_df_street = (
  spark.read
    .option("multiline", "true")
    .format("json")
    .load(f"{etl_dir}/nyc_streets.geojson")
      .select("type", F.explode(col("features")).alias("feature"))
      .select("type", col("feature.properties").alias("properties"), F.to_json(col("feature.geometry")).alias("json_geometry"))
    .withColumn("geom", mos.st_geomfromgeojson("json_geometry"))
    .withColumn("geom", mos.st_setsrid("geom", F.lit(4326)))
    .withColumn("geom_wkt", mos.st_astext("geom"))
    .withColumn("is_valid", mos.st_isvalid("geom"))
    .withColumn("is_coords_4326", mos.st_hasvalidcoordinates("geom", F.lit("EPSG:4326"), F.lit('bounds')))
    .select("properties.*", "geom_wkt", "is_valid", "is_coords_4326", "geom")
    .withColumn(
      "borough_name", 
      F.when(col("boroughcod") == 1, "Manhattan")
      .when(col("boroughcod") == 2, "Bronx")
      .when(col("boroughcod") == 3, "Brooklyn")
      .when(col("boroughcod") == 4, "Queens")
      .when(col("boroughcod") == 5, "Staten Island")
      .otherwise("UKN")
    )  
    .withColumn(
      "county", 
      F.when(col("boroughcod") == 1, "New York")
      .when(col("boroughcod") == 2, "Bronx")
      .when(col("boroughcod") == 3, "Kings")
      .when(col("boroughcod") == 4, "Queens")
      .when(col("boroughcod") == 5, "Richmond")
      .otherwise("UKN")
    )  
    .withColumn("zipcode", col("l_zip"))
    .withColumn("name", col("stname_lab"))
    .withColumn("row_id", F.monotonically_increasing_id())
    .selectExpr(
      "row_id", "borough_name", "zipcode", "name", "county", "geom_wkt",
                "* except(row_id, borough_name, zipcode, name, county, geom_wkt)"
    )
)
_df_street.display()

# COMMAND ----------

# MAGIC %md ### Write to Delta
# MAGIC 
# MAGIC > Results in `street` table

# COMMAND ----------

df_street = write_to_delta(_df_street, f"{etl_dir}/street", "street", db_name, zorder_col='zipcode', register_table=True, overwrite=False)
print(f"count? {df_street.count():,}")
df_street.display()

# COMMAND ----------

df_street_kepler = df_street.select("name", "geom_wkt") # <-- reduce num of field to aid in rendering

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df_street_kepler "geom_wkt" "geometry" 120_000

# COMMAND ----------

#%sql show tables
