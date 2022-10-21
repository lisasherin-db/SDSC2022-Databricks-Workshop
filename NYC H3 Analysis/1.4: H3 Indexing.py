# Databricks notebook source
# MAGIC %md # H3 Indexing
# MAGIC 
# MAGIC <img src="https://camo.githubusercontent.com/7d659087f179cb105dd0b91854bcc00d7b0a15d61cdac0e3d2afe05993dc63a7/68747470733a2f2f756265722e6769746875622e696f2f696d672f68334c6f676f2d636f6c6f722e737667" width="10%"/>
# MAGIC 
# MAGIC > Use Built-In [H3 Product APIs](https://docs.databricks.com/spark/latest/spark-sql/language-manual/sql-ref-functions-builtin.html#h3-geospatial-functions) to prepare Delta tables with H3 indexes for the Streets (LINE), Stores (POINT), and Neighborhood Tabulation Areas (POLYGON) for existing NYC Delta base tables prepared in the prior Notebooks.
# MAGIC 
# MAGIC __Produces the following Tables:__
# MAGIC <p/>
# MAGIC 
# MAGIC * `nta_pop_cell`
# MAGIC * `nta_pop_cell_compact`
# MAGIC * `store_cell`
# MAGIC * `street_cell`
# MAGIC * `street_cell_agg`
# MAGIC 
# MAGIC --- 
# MAGIC __Author: Michael Johns <mjohns@databricks.com>__  
# MAGIC _Last Modified: 17 OCT 2022_

# COMMAND ----------

# MAGIC %md ## Setup
# MAGIC <p/>
# MAGIC 
# MAGIC 1. Import Databricks H3 columnar functions for DBR / DBSQL Photon with `from pyspark.databricks.sql.functions import *`
# MAGIC 1. To use [H3 OSS](https://h3geo.org/) library standalone or within User Defined Functions (UDF) for `h3_line`:
# MAGIC   * Install with `%pip install h3==3.7.0` [[1](https://docs.databricks.com/libraries/notebooks-python-libraries.html#manage-libraries-with-pip-commands) | [2](https://docs.databricks.com/release-notes/runtime/11.2.html#installed-java-and-scala-libraries-scala-212-cluster-version)] to match built-in native and JAR (resets python context)
# MAGIC   * Import with `import h3 as h3lib`
# MAGIC 1. To use [Kepler.gl](https://kepler.gl/) OSS library for map layer rendering:
# MAGIC   * Install with `%pip install keplergl`
# MAGIC   * Import with `from keplergl import KeplerGl`
# MAGIC 
# MAGIC __Note: If you hit `H3_NOT_ENABLED` [[docs](https://docs.databricks.com/error-messages/h3-not-enabled-error-class.html#h3_not_enabled-error-class)]__
# MAGIC 
# MAGIC > `h3Expression` is disabled or unsupported. Consider enabling Photon or switch to a tier that supports H3 expressions. [[AWS](https://www.databricks.com/product/aws-pricing) | [Azure](https://azure.microsoft.com/en-us/pricing/details/databricks/) | [GCP](https://www.databricks.com/product/gcp-pricing)]
# MAGIC 
# MAGIC __Recommend running on DBR 11.3+ for maximizing photonized functions.__

# COMMAND ----------

# MAGIC %pip install h3==3.7.0 keplergl --quiet

# COMMAND ----------

from pyspark.databricks.sql.functions import *

from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.sql import Window

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

# MAGIC %sql show tables

# COMMAND ----------

h3_res = 12

# COMMAND ----------

# MAGIC %md ## `street`

# COMMAND ----------

df_street = spark.table("street")
display(df_street.limit(3))

# COMMAND ----------

@udf(returnType=ArrayType(LongType()))
def h3_line_arr(c1, c2):
  """
  Get line of cells between two cells.
  - accepts str or long
  - returns long array.
  - this uses OSS h3lib
  """
  import h3 as h3lib
  
  if c1 is None and c2 is None:
    return []
  elif c1 is None:
    if isinstance(c2, str):
      return [h3lib.string_to_h3(c2)]
    return [c2]
  elif c2 is None:
    if isinstance(c1, str):
      return [h3lib.string_to_h3(c1)]
    return [c1]
  
  _c1 = c1
  _c2 = c2
  if not isinstance(c1, str):
    _c1 = h3lib.h3_to_string(c1)
    _c2 = h3lib.h3_to_string(c2)  
    
  cs = h3lib.h3_line(_c1,_c2)
  return [h3lib.string_to_h3(c) for c in cs]

spark.udf.register("h3_line_arr", h3_line_arr) # <-- register for SQL

# COMMAND ----------

# MAGIC %md __H3 Index Line Segments__
# MAGIC 
# MAGIC 1. Explode each Line Coord -- `F.posexplode("line_coords")`
# MAGIC 1. Lag Windox for identifying Segments -- `withColumn("cell_lag", F.lag('cell', 1).over(lag_window))`
# MAGIC 1. Built-In H3 Columnar function for cell -- `h3_longlatash3(col("point_coord")[0], col("point_coord")[1], h3_res)`
# MAGIC 1. UDF using H3 OSS lib to get H3 cells for Line Segments -- `withColumn("h3_line", h3_line_arr(col("cell_lag"),col("cell")))`

# COMMAND ----------

lag_window = Window.partitionBy('row_id').orderBy('line_pos')

_df_street_cell_lag = (
  df_street
  .withColumn("line_coords", col("geom")['boundary'][0])
  .withColumn("num_line_coords", F.size("line_coords"))
    .select("row_id", "borough_name", "zipcode", "name", "physicalid", "shape_leng", "num_line_coords", F.posexplode("line_coords"))
    .selectExpr("* except (col, pos)", "pos as line_pos", "col as point_coord")
  .withColumn("cell", h3_longlatash3(col("point_coord")[0], col("point_coord")[1], h3_res))
  .withColumn("cell_lag", F.lag('cell', 1).over(lag_window))
    .filter("line_pos > 0") # <-- 0 has no lag
  .withColumn("h3_line", h3_line_arr(col("cell_lag"),col("cell")))
    .selectExpr("* except(point_coord, cell, cell_lag, h3_line)", "cell_lag as cell_p1", "cell as cell_p2", "h3_line")
)
# display(_df_street_cell_lag)

# COMMAND ----------

# MAGIC %md __Steet Cell Aggregate__
# MAGIC 
# MAGIC 1. Built-In columnar function for string id of cell (useful for Kepler rendering and H3 OSS lib) `h3_h3tostring("cell")`
# MAGIC 1. Aggregate `cells` and `cells_str` into sets (for lists / routing will use non-aggregated table):
# MAGIC   ```
# MAGIC   .groupBy("row_id")
# MAGIC         .agg(
# MAGIC           F.collect_set("cell").alias("cells"), 
# MAGIC           F.collect_set("cell_str").alias("cells_str")
# MAGIC         )
# MAGIC   ```

# COMMAND ----------

_df_street_cell_agg = (
  df_street
    .join(
      _df_street_cell_lag
        .select("row_id", F.explode("h3_line").alias("cell"))
        .withColumn("cell_str", h3_h3tostring("cell"))
      .groupBy("row_id")
        .agg(
          F.collect_set("cell").alias("cells"), 
          F.collect_set("cell_str").alias("cells_str")
        )
        .select("row_id", F.lit(h3_res).alias("h3_res"), F.size("cells").alias("num_cells"), "cells", "cells_str"),
      ["row_id"]
    )
    .orderBy("row_id")
)
display(_df_street_cell_agg)

# COMMAND ----------

# MAGIC %md ### Write to Delta
# MAGIC 
# MAGIC > Results in `street_cell_agg` as well as `street_cell` tables

# COMMAND ----------

df_street_cell_agg = write_to_delta(_df_street_cell_agg, f"{etl_dir}/street_cell_agg", "street_cell_agg", db_name, zorder_col='row_id', register_table=True, overwrite=False)
print(f"count? {df_street_cell_agg.count():,}")
df_street_cell_agg.display()

# COMMAND ----------

display(df_street_cell_agg.groupBy("num_cells").count().orderBy("num_cells"))

# COMMAND ----------

# MAGIC %md __Persist Exploded version of Street Cells__
# MAGIC 
# MAGIC _Note: This is z-ordered by cell when written out to Delta for optimizing index filters and joins_

# COMMAND ----------

_df_street_cell = (
  df_street_cell_agg
    .selectExpr("* except(cells, cells_str)", "explode(cells) as cell")
    .withColumn("cell_str", h3_h3tostring("cell"))
    .selectExpr("cell", "* except(cell)")
)
display(_df_street_cell)

# COMMAND ----------

df_street_cell = write_to_delta(_df_street_cell, f"{etl_dir}/street_cell", "street_cell", db_name, zorder_col='cell', register_table=True, overwrite=False)
print(f"count? {df_street_cell.count():,}")
df_street_cell.display()

# COMMAND ----------

# MAGIC %md ## `store`

# COMMAND ----------

df_store = spark.table("store")
display(df_store.limit(3))

# COMMAND ----------

# MAGIC %md __Store Cell__
# MAGIC 1. Built-In H3 Columnar function for cell `h3_longlatash3("point_x", "point_y", h3_res)`
# MAGIC 1. Built-In H3 Columnar function for cell string (useful for Kepler and H3 OSS) `h3_longlatash3string("point_x", "point_y", h3_res)`
# MAGIC 
# MAGIC _Note: This is z-ordered by cell when written out to Delta for optimizing index filters and joins)_

# COMMAND ----------

_df_store_cell = (
  df_store
    .withColumn("h3_res", F.lit(h3_res))
    .withColumn("cell", h3_longlatash3("point_x", "point_y", h3_res))
    .withColumn("cell_str", h3_longlatash3string("point_x", "point_y", h3_res))
    .selectExpr("cell", "* except(cell, cell_str, h3_res)", "h3_res", "cell_str")
    .filter(col("cell").isNotNull())
)
display(_df_store_cell)

# COMMAND ----------

# MAGIC %md ### Write to Delta
# MAGIC 
# MAGIC > Results in `store_cell` table

# COMMAND ----------

df_store_cell = write_to_delta(_df_store_cell, f"{etl_dir}/store_cell", "store_cell", db_name, zorder_col='cell', register_table=True, overwrite=False)
print(f"count? {df_store_cell.count():,}")
df_store_cell.display()

# COMMAND ----------

# MAGIC %md ## `nta_pop`

# COMMAND ----------

df_nta_pop = spark.table("nta_pop")
display(df_nta_pop.limit(1))

# COMMAND ----------

# MAGIC %md __NTA Population Cell Compact__
# MAGIC 1. Built-In H3 Columnar functions `h3_compact(h3_polyfillash3("geom_wkt", h3_res))`:
# MAGIC   * Polyfill at specified H3 resolution (in this case 12)
# MAGIC   * Compact those cells (will various combinations of res <= 12) -- this allows smaller data / arrays to encapsulate the same information, can then be uncompacted with `h3_uncompact` at a resolution >= the compacted resolution, so in this case 12 to 15.

# COMMAND ----------

_df_nta_pop_cell_compact = (
  df_nta_pop
    .withColumn("h3_res", F.lit(h3_res))
    .withColumn("cells_compact", h3_compact(h3_polyfillash3("geom_wkt", h3_res)))
)

display(_df_nta_pop_cell_compact)

# COMMAND ----------

# MAGIC %md ### Write to Delta
# MAGIC 
# MAGIC > Results in `nta_pop_cell_compact` and `nta_pop_cell` tables

# COMMAND ----------

df_nta_pop_cell_compact = write_to_delta(_df_nta_pop_cell_compact, f"{etl_dir}/nta_pop_cell_compact", "nta_pop_cell_compact", db_name, zorder_col='ntacode', register_table=True, overwrite=False)
print(f"count? {df_nta_pop_cell_compact.count():,}")
df_nta_pop_cell_compact.display()

# COMMAND ----------

# MAGIC %md __NTA Population Cell__
# MAGIC 1. Built-In H3 Columnar function to uncompact at resolution 12 `F.explode(h3_uncompact("cells_compact", "h3_res"))`:
# MAGIC 
# MAGIC _Note: This is z-ordered by cell when written out to Delta for optimizing index filters and joins_

# COMMAND ----------

_df_nta_pop_cell = (
  df_nta_pop_cell_compact
    .drop("geom", "geom_wkt")
    .select("*", F.explode(h3_uncompact("cells_compact", "h3_res")).alias("cell"))
    .drop("cells_compact")
    .selectExpr("cell", "* except(cell)", "h3_h3tostring(cell) as cell_str") # <-- using sql api
)
print(f"count? {_df_nta_pop_cell.count():,}")
# _df_nta_pop_cell.display()

# COMMAND ----------

df_nta_pop_cell = write_to_delta(_df_nta_pop_cell, f"{etl_dir}/nta_pop_cell", "nta_pop_cell", db_name, zorder_col='cell', register_table=True, overwrite=False)
print(f"count? {df_nta_pop_cell.count():,}")
df_nta_pop_cell.display()

# COMMAND ----------

# MAGIC %md ## Render in Kepler
# MAGIC 
# MAGIC > Just to see the combination of the H3 tables generated in this notebook.

# COMMAND ----------

county = 'New York' 
# e.g. for Manhatten / New York county
ntacodes = [
  'MN25', # Battery Park
  'MN27', # Chinatown
  'MN24', # SoHo
  'MN28'  # Lower East
]
map_all = KeplerGl(height=600, config={'mapState': {'latitude': 40.7, 'longitude': -74.0, 'zoom': 13}})
map_all.add_data(data=df_store.drop("geom").filter(f"county == '{county}'").toPandas(), name="store")
map_all.add_data(data=df_store_cell.select("cell_str", "zip_code", "county").filter(f"county == '{county}'").toPandas(), name=f"store_h3_res_{h3_res}")
map_all.add_data(data=df_street_cell.select("cell_str", "county", "zipcode", "name").filter(f"county == '{county}'").toPandas(), name=f"street_h3_res_{h3_res}")
map_all.add_data(data=df_nta_pop_cell.select("cell_str", "county", "ntaname", "pop_year_2010", "pop_year_2000", "ntacode").filter(col("ntacode").isin(*ntacodes)).toPandas(), name=f"nta_pop_h3_res_{h3_res}")
display_kepler(map_all)

# COMMAND ----------

# MAGIC %sql show tables
