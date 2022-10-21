# Databricks notebook source
# MAGIC %md # NYC Retail Food Stores
# MAGIC 
# MAGIC > A listing of all retail food stores which are licensed by the Department of Agriculture and Markets [[1](https://data.ny.gov/Economic-Development/Retail-Food-Stores-Map/p2dn-xhaw)]
# MAGIC 
# MAGIC * CSV from https://data.ny.gov/api/views/9a8c-vfzj/rows.csv?accessType=DOWNLOAD&sorting=true
# MAGIC * Data Catalog at https://data.ny.gov/Economic-Development/Retail-Food-Stores/9a8c-vfzj
# MAGIC 
# MAGIC __ESTABLISHMENT CODES__
# MAGIC |   |   | 
# MAGIC |---|---|
# MAGIC | A(3)–Store | M – Salvage Dealer |
# MAGIC | B – Bakery | N – Wholesale Produce Packer |
# MAGIC | C – Food Manufacturer | O – Produce Grower/Packer/Broker, Storage |
# MAGIC | D – Food Warehouse | P – C.A. Room |
# MAGIC | E – Beverage Plant | Q – Feed Mill/Medicated |
# MAGIC | F – Feed Mill/Non-Medicated | R – Pet Food Manufacturer |
# MAGIC | G - Processing Plant | S – Feed Warehouse and/or Distributor |
# MAGIC | H - Wholesale Manufacturer | T – Disposal Plant |
# MAGIC | I - Refrigerated Warehouse | U - Disposal Plant/Transportation Service |
# MAGIC | J – Multiple Operations | V – Slaughterhouse |
# MAGIC | K - Vehicle | W – Farm Winery-Exempt |
# MAGIC | L - Produce Refrigerated Warehouse | Z - Farm Product Use Only |
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

from keplergl import KeplerGl
def display_kepler(kmap:KeplerGl, height=800, width=1200) -> None:
  """
  Convenience function to render map in kepler.gl
  - use this when cannot directly render or
    want to go beyond the %%mosaic_kepler magic.
  """
  displayHTML(
    kmap
      ._repr_html_()
      .decode("utf-8")
      .replace(".height||400", f".height||{height}")
      .replace(".width||400", f".width||{width}")
  )

# COMMAND ----------

# MAGIC %run ./Helpers/nyc_data_config.py

# COMMAND ----------

# MAGIC %run ./Helpers/nyc_table_helper.py

# COMMAND ----------

# MAGIC %md ### Download NY Retail (1x)

# COMMAND ----------

#%sh 
#wget -O ny_retail.csv "https://data.ny.gov/api/views/9a8c-vfzj/rows.csv?accessType=DOWNLOAD&sorting=true"
#mkdir -p $etl_dir_fuse
#cp ny_retail.csv $etl_dir_fuse/

# COMMAND ----------

display(dbutils.fs.ls(etl_dir))

# COMMAND ----------

# MAGIC %md __Check Data__

# COMMAND ----------

# display(
#   spark
#     .read
#     .csv(f"{etl_dir}/ny_retail.csv", header=True, inferSchema=True)
#     .limit(10)
# )

# COMMAND ----------

# MAGIC %md ## Load Stores
# MAGIC 
# MAGIC > Will use built-in CSV reader
# MAGIC 
# MAGIC __Mosaic Functions__
# MAGIC 
# MAGIC ```
# MAGIC .withColumn("geom", mos.st_geomfromwkt("geom_wkt"))
# MAGIC .withColumn("geom", mos.st_setsrid("geom", F.lit(4326)))
# MAGIC .withColumn("is_valid", mos.st_isvalid("geom"))
# MAGIC .withColumn("is_coords_4326", mos.st_hasvalidcoordinates("geom", F.lit("EPSG:4326"), F.lit('bounds')))
# MAGIC ```

# COMMAND ----------

_df_store = (
  spark
    .read
    .csv(f"{etl_dir}/ny_retail.csv", header=True, inferSchema=True)
    .withColumnRenamed("georeference", "geom_wkt")
    .withColumn("geom", mos.st_geomfromwkt("geom_wkt"))
    .withColumn("geom", mos.st_setsrid("geom", F.lit(4326)))
    .withColumn("is_valid", mos.st_isvalid("geom"))
    .withColumn("is_coords_4326", mos.st_hasvalidcoordinates("geom", F.lit("EPSG:4326"), F.lit('bounds')))
    .withColumn("point_x", col("geom")['boundary'][0][0][0])
    .withColumn("point_y", col("geom")['boundary'][0][0][1])
)

_df_store = _df_store.toDF(*[c.lower().replace(' ','_') for c in _df_store.columns])

# display(_df_store)

# COMMAND ----------

# MAGIC %md ### Write to Delta
# MAGIC 
# MAGIC > Results in `store` table

# COMMAND ----------

df_store = write_to_delta(_df_store, f"{etl_dir}/store", "store", db_name, zorder_col='zip_code', register_table=True, overwrite=False)
print(f"count? {df_store.count():,}")
df_store.display()

# COMMAND ----------

# MAGIC %md __Render the Stores in NYC__
# MAGIC 
# MAGIC > Also showing H3 cell at resolution 10 and NYC streets from previous notebook.

# COMMAND ----------

# -- Using product built-in columnar H3 functions for store cells --
h3_resolution = 10
df_store_h3 = (
  df_store
    .withColumn("cell_id", h3_longlatash3("point_x", "point_y", h3_resolution))
    .withColumn("cell_id_str", h3_longlatash3string("point_x", "point_y", h3_resolution))
    .selectExpr("cell_id", "* except(cell_id, cell_id_str)",  "cell_id_str")
    .filter(col("cell_id").isNotNull())
)
display(df_store_h3)

# COMMAND ----------

# MAGIC %md __Zoom-In for better details__

# COMMAND ----------

map_1 = KeplerGl(height=600, config={'mapState': {'latitude': 40.74, 'longitude': -73.95, 'zoom': 12}})
map_1.add_data(data=df_store.drop("geom").filter("county == 'New York'").toPandas(), name="store")
map_1.add_data(data=df_store_h3.select("cell_id_str", "zip_code", "county").filter("county == 'New York'").toPandas(), name=f"store_h3_res_{h3_resolution}")
map_1.add_data(data=spark.table("street").select("county", "zipcode", "name", "borough_name", "geom_wkt").filter("county == 'New York'").toPandas(), name="streets")
display_kepler(map_1)

# COMMAND ----------

# %sql show tables
