# Databricks notebook source
# MAGIC %md # H3 Proximity Analysis (Python)
# MAGIC 
# MAGIC <img src="https://camo.githubusercontent.com/7d659087f179cb105dd0b91854bcc00d7b0a15d61cdac0e3d2afe05993dc63a7/68747470733a2f2f756265722e6769746875622e696f2f696d672f68334c6f676f2d636f6c6f722e737667" width="10%"/>
# MAGIC 
# MAGIC > Use Built-In [H3 Product APIs](https://docs.databricks.com/spark/latest/spark-sql/language-manual/sql-ref-functions-builtin.html#h3-geospatial-functions) to analyze distance of each h3 cell of interest to resource of interest, in this case food stores. Will consider both straight line distance and road distance.
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
# MAGIC _Last Modified: 18 OCT 2022_

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

import h3 as h3lib # <-- want to use product where possible

import matplotlib.pyplot as plt
import matplotlib.image as img
plt.rcParams["figure.figsize"] = [8, 6] # <-- inches
plt.rcParams["figure.autolayout"] = True

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

# MAGIC %sql show tables

# COMMAND ----------

# MAGIC %md ### Load all the H3 Cell Tables
# MAGIC 
# MAGIC > Since we have previously prepared all tables with resolution 12, they are ready for our analysis

# COMMAND ----------

h3_res = 12

# COMMAND ----------

df_store_cell = spark.table("store_cell")
display(df_store_cell.limit(3))

# COMMAND ----------

df_street_cell = spark.table("street_cell")
display(df_street_cell.limit(3))

# COMMAND ----------

df_nta_pop_cell = spark.table("nta_pop_cell")
display(df_nta_pop_cell.limit(3))

# COMMAND ----------

# MAGIC %md ## Assess Approximate H3 Cell Size for NYC
# MAGIC 
# MAGIC > H3 cells are projected onto a sphere so size is not uniform depending on the latitude of the cell as well as other various distortions that are unavoidable. For NYC we see that H3 size is approximately __18.3m or 60ft__ between neighbor cell centerpoints at __Resolution 12__. We will use this as the basis for our K-Ring and Hex Ring analysis.

# COMMAND ----------

display(
  df_store_cell
    .select(
      "cell","cell_str",
      h3_centerasgeojson("cell").alias("cell_center"),
      h3_kring("cell", 1).alias("cell_kring_1")
    )
    .withColumn("neighbor", col("cell_kring_1")[1])
    .withColumn("neighbor_str", h3_h3tostring("neighbor"))
    .withColumn("neighbor_center", h3_centerasgeojson("neighbor"))
    .selectExpr("cell", "cell_str", "cell_center", "neighbor", "neighbor_str", "neighbor_center")
    .limit(1)
)

# COMMAND ----------

# MAGIC %md __Use H3 OSS lib [point_dist](https://h3geo.org/docs/3.x/api/misc#pointdistm) to get distance between 2 neighbor cells__
# MAGIC 
# MAGIC > Gives the "great circle" or "haversine" distance between pairs of GeoCoord points (lat/lng pairs) in meters, kilometers, or radians (defaults to kilometers).

# COMMAND ----------

print(f"""approximate distance in meters? {h3lib.point_dist(h3lib.h3_to_geo("8c2a154e5376bff"), h3lib.h3_to_geo("8c2a154e53769ff"), unit='m'):,}""")
print(f"""approximate distance in kilometers? {h3lib.point_dist(h3lib.h3_to_geo("8c2a154e5376bff"), h3lib.h3_to_geo("8c2a154e53769ff"), unit='km'):,}""")
print(f"""approximate distance in radians? {h3lib.point_dist(h3lib.h3_to_geo("8c2a154e5376bff"), h3lib.h3_to_geo("8c2a154e53769ff"), unit='rads'):,}""")

# COMMAND ----------

# MAGIC %md ## Food Store H3 K-Rings
# MAGIC 
# MAGIC > Use K-Rings of values up to 20 for access to stores in increments of 5. _This equates to straight line distances of up to 1200 ft or just under 1/4 mile at 20 k-rings and 1/8 mile at 10 k-rings._
# MAGIC 
# MAGIC __Number Cells per Ring__
# MAGIC <p/>
# MAGIC 
# MAGIC * `K-Ring = 05`: 91 cells
# MAGIC * `K-Ring = 10`: 331 cells
# MAGIC * `K-Ring = 15`: 721 cells
# MAGIC * `K-Ring = 20`: 1,261 cells
# MAGIC 
# MAGIC __Note: We filter by NYC counties, yielding 13.4K Food Stores.__

# COMMAND ----------

nyc_counties = [r[0] for r in df_street_cell.select("county").distinct().collect()]
nyc_counties

# COMMAND ----------

df_store_kring_agg = (
  df_store_cell
    .filter(col("cell").isNotNull())
    .withColumn("k5", h3_kring("cell",   5))
    .withColumn("k10", h3_kring("cell", 10))
    .withColumn("k15", h3_kring("cell", 15))
    .withColumn("k20", h3_kring("cell", 20))
    .withColumn("num_k5", F.size("k5"))
    .withColumn("num_k10", F.size("k10"))
    .withColumn("num_k15", F.size("k15"))
    .withColumn("num_k20", F.size("k20"))
  .filter(col("county").isin(nyc_counties))
  .selectExpr(
    "cell", "county", "zip_code", "entity_name", "num_k5", "num_k10", "num_k15", "num_k20",
    "* except(cell, county, entity_name, num_k5, num_k10, num_k15, num_k20)"
  )
)

print(f"count? {df_store_kring_agg.count():,}")
display(df_store_kring_agg.limit(1))

# COMMAND ----------

# MAGIC %md __Explode K-Ring cells for 1 cell per row__

# COMMAND ----------

df_k5 = (
  df_store_kring_agg
  .selectExpr("cell as store_cell", "explode(k5) as k_cell", "5 as ring_k", "county", "zip_code", "entity_name", "establishment_type", "license_number")
)

df_k10 = (
  df_store_kring_agg
  .selectExpr("cell as store_cell", "explode(k10) as k_cell", "10 as ring_k", "county", "zip_code", "entity_name", "establishment_type", "license_number")
)

df_k15 = (
  df_store_kring_agg
  .selectExpr("cell as store_cell", "explode(k15) as k_cell", "15 as ring_k", "county", "zip_code", "entity_name", "establishment_type", "license_number")
)

df_k20 = (
  df_store_kring_agg
  .selectExpr("cell as store_cell", "explode(k20) as k_cell", "20 as ring_k", "county", "zip_code", "entity_name", "establishment_type", "license_number")
)

print(f"count? {df_k5.count():,}")
display(df_k5.limit(1))

# COMMAND ----------

# MAGIC %md __Render K5 and K10__
# MAGIC 
# MAGIC > For the given zip code, seee that coverage has some gaps but not too many (just showing rings for less overlap).
# MAGIC 
# MAGIC _Screenshot included for reading outside Databricks and to show some additional rendering tweaks._

# COMMAND ----------

plt.imshow(img.imread("Resources/store_hex_ring_5_10.png"), aspect='auto')
plt.show()

# COMMAND ----------

# -- financial district zip codes are 10004, 10005, 10006, 10007, 10038 --
# See https://bklyndesigns.com/new-york-city-zip-code/ for additional 
# - show hex rings for space savings
zip_code = 10451

df_hr5_10_kepler = (
  df_k5
    .withColumn("hr_cells", h3_hexring("store_cell", "ring_k"))
    .withColumn("hr_cell", F.explode("hr_cells"))
    .select("hr_cell", "ring_k", "zip_code", "establishment_type")
  .union(
    df_k10
      .withColumn("hr_cells", h3_hexring("store_cell", "ring_k"))
      .withColumn("hr_cell", F.explode("hr_cells"))
      .select("hr_cell", "ring_k", "zip_code", "establishment_type")
  )
  .withColumn("hr_cell_str", h3_h3tostring("hr_cell"))
  .filter(f"zip_code == {zip_code}")
  .filter(
    F.lower("establishment_type")
      .contains("ja") # multiple locations, storefront
  )
  .distinct()
)
df_hr5_10_kepler.count()

# COMMAND ----------

display_kepler(
  KeplerGl(
    config={ 
      'version': 'v1', 
      'mapState': {
        'latitude': 40.82, 
        'longitude': -73.92, 
        'zoom': 13.5
      }, 
      'mapStyle': {'styleType': 'dark'},
      'options': {'readOnly': False, 'centerMap': True}
    },
    data={
      'hexring_5_10': df_hr5_10_kepler.toPandas()
    },
    show_docs=False,
  )
)

# COMMAND ----------

# MAGIC %md __Subtract K-Ring Cells out at each of the given Ring Levels__
# MAGIC 
# MAGIC > This yield gap areas that are not accessible at a given k-ring level.
# MAGIC 
# MAGIC ```
# MAGIC count all NTA cells? 2,536,715
# MAGIC ...cells outside K-Ring=5 areas? 2,013,961
# MAGIC ...cells outside K-Ring=10 areas? 1,498,437
# MAGIC ...cells outside K-Ring=15 areas? 1,147,727
# MAGIC ...cells outside K-Ring=20 areas? 898,175
# MAGIC ```

# COMMAND ----------

df_no_k5 =(
  df_nta_pop_cell.select("cell")
    .subtract(df_k5.selectExpr("k_cell as cell"))
)

df_no_k10 =(
  df_nta_pop_cell.select("cell")
    .subtract(df_k10.selectExpr("k_cell as cell"))
)

df_no_k15 =(
  df_nta_pop_cell.select("cell")
    .subtract(df_k15.selectExpr("k_cell as cell"))
)

df_no_k20 =(
  df_nta_pop_cell.select("cell")
    .subtract(df_k20.selectExpr("k_cell as cell"))
)

df_no = (
  df_nta_pop_cell
    .join(df_no_k5.selectExpr("cell","False as in_kring_5"),["cell"],"left_outer")
      .withColumn("in_kring_5", F.when(col("in_kring_5").isNull(), True).otherwise(False))
    .join(df_no_k10.selectExpr("cell","False as in_kring_10"),["cell"],"left_outer")
      .withColumn("in_kring_10", F.when(col("in_kring_10").isNull(), True).otherwise(False))
    .join(df_no_k15.selectExpr("cell","False as in_kring_15"),["cell"],"left_outer")
      .withColumn("in_kring_15", F.when(col("in_kring_15").isNull(), True).otherwise(False))
    .join(df_no_k20.selectExpr("cell","False as in_kring_20"),["cell"],"left_outer")
      .withColumn("in_kring_20", F.when(col("in_kring_20").isNull(), True).otherwise(False))
)

print(f"count all NTA 'no' cells? {df_no.count():,}")
print(f"""... no_kring_5? {df_no.filter("in_kring_5 == False").count():,}""")
print(f"""... no_kring_10? {df_no.filter("in_kring_10 == False").count():,}""")
print(f"""... no_kring_15? {df_no.filter("in_kring_15 == False").count():,}""")
print(f"""... no_kring_20? {df_no.filter("in_kring_20 == False").count():,}""")

display(df_no.limit(3))

# COMMAND ----------

# MAGIC %md __Which NTAs have the greatest number of cells outside the food store k-rings?__
# MAGIC 
# MAGIC > Limiting to 25 for visualization purposes. See that `QN99`, `SI05`, `QN98`, `BK99`, and `BX99` round out the Top (or Worst) 5. 
# MAGIC 
# MAGIC _Again, screenshot included for reading outside Databricks and to show some additional rendering tweaks._

# COMMAND ----------

plt.figure(figsize=(15, 5))
plt.imshow(img.imread("Resources/lowest_kring_20_cells.png"), aspect='auto')
plt.show()

# COMMAND ----------

display(
  df_no
    .filter("in_kring_5 == False")
    .groupBy("ntacode")
      .count()
    .selectExpr("ntacode", "count as count_no_k5")
  .join(
    df_no
      .filter("in_kring_10 == False")
      .groupBy("ntacode")
        .count()
      .selectExpr("ntacode", "count as count_no_k10"),
    ["ntacode"]
  )
  .join(
    df_no
      .filter("in_kring_15 == False")
      .groupBy("ntacode")
        .count()
      .selectExpr("ntacode", "count as count_no_k15"),
    ["ntacode"]
  )
  .join(
    df_no
      .filter("in_kring_20 == False")
      .groupBy("ntacode")
        .count()
      .selectExpr("ntacode", "count as count_no_k20"),
    ["ntacode"]
  )
    .orderBy(F.desc("count_no_k20"))
    .limit(25)
)

# COMMAND ----------

# MAGIC %md __Render For Cells Outside of K20__
# MAGIC 
# MAGIC > Hint: you can turn on | off layers in Kepler as well as mess with colors and fills after rendering.
# MAGIC 
# MAGIC _Again, screenshot included for reading outside Databricks and to show some additional rendering tweaks._

# COMMAND ----------

plt.imshow(img.imread("Resources/store_kring_20_gaps.png"), aspect='auto')
plt.show()

# COMMAND ----------

county = "Queens"    #also Bronx Queens
ntacodes = ['QN99'] #also 'BX22'

pdf_store = spark.table("store").select("geom_wkt", "entity_name", "county", "establishment_type", "license_number",).filter(f"county == '{county}'").toPandas()
pdf_store_cell = df_store_cell.select("cell_str", "zip_code", "county").filter(col("cell_str").isNotNull()).filter(f"county == '{county}'").toPandas()
pdf_nta_no = df_no.filter("in_kring_20 == False").select("cell_str", "ntacode", "ntaname").filter(col("ntacode").isin(*ntacodes)).toPandas()
pdf_nta_geom = spark.table("nta_pop").select("geom_wkt", "county", "ntaname", "pop_year_2010", "pop_year_2000", "ntacode").filter(col("ntacode").isin(*ntacodes)).toPandas()

# -- Further Layer Tweaks --
# - For `no_k20_cell` set color to red
# - turn off fill for nta_pop geom_wkt
map_nta_config={
  'version': 'v1', 
  'mapState': {
    'latitude': 40.675544649494356, 
    'longitude': -73.81784574884483, 
    'zoom': 10.631611941798617
  }, 
  'mapStyle': {'styleType': 'satellite'},
  "options": {"readOnly": False, "centerMap": True}
} 

display_kepler(
  KeplerGl(
    config=map_nta_config,
    data={
      'store': pdf_store,
      'store_cell': pdf_store_cell,
      'no_k20_cell': pdf_nta_no,
      'nta_pop': pdf_nta_geom
    },
    show_docs=False
  )
)

# COMMAND ----------

# MAGIC %md ## Food Store H3 K-Rings with Streets 
# MAGIC 
# MAGIC > For each street cell determine if it is within K-Ring 20 of any store. This allows more isolation to what street locations might benefit from the addition of a food store.

# COMMAND ----------

# MAGIC %md _Generate DataFrame with __ALL 18.7M__ k-ring=20 (1/4 mile) street cells identified; also has non-streets and those outside the k-ring._

# COMMAND ----------

df_k20_street = (
  df_store_kring_agg
    .filter(col("cell").isNotNull()) # <-- store_cell
    .selectExpr("cell as store_cell", "True as is_k20", "cell_str as store_cell_str", "zip_code", "entity_name", "establishment_type", "license_number", "k20") 
    .selectExpr("explode(k20) as cell", "*")
    .withColumn("k20_cell_str", h3_h3tostring("cell"))
      .drop("k20")
  .join(
    df_street_cell
      .selectExpr("cell", "True as is_street_cell", "name as street_name",  "row_id as street_row_id", "cell_str as street_cell_str"),
    ["cell"],
    "full_outer"
  )
  .join(
    df_nta_pop_cell
      .withColumnRenamed("cell_str","nta_cell_str"),
    ["cell"],
    "left_outer"
  )
  .withColumn("is_street_cell", F.when(col("is_street_cell").isNull(), False).otherwise(True))
  .withColumn("is_k20", F.when(col("is_k20").isNull(), False).otherwise(True))
  .withColumn("nbool_k20", col("is_k20").cast(IntegerType())) # <-- for layer coloring
)

print(f"count? {df_k20_street.count():,}")
display(df_k20_street.limit(1))

# COMMAND ----------

# MAGIC %md __Which NTAs have the fewest number of street cells within the k-ring limits?__
# MAGIC 
# MAGIC > Showing Top (or Worst) 50 for rendering.
# MAGIC 
# MAGIC _Again, screenshot included for reading outside Databricks and to show some additional rendering tweaks._

# COMMAND ----------

plt.figure(figsize=(15, 5))
plt.imshow(img.imread("Resources/lowest_kring_20_streets.png"), aspect='auto')
plt.show()

# COMMAND ----------

display(
  df_k20_street
    .filter("is_street_cell == True")
    .filter("is_k20 == True")
    .groupBy("ntacode", "county")
    .count()
    .orderBy("count")
    .limit(50)
)

# COMMAND ----------

# MAGIC %md __Let's look at NTA `BX22` in the Bronx__
# MAGIC 
# MAGIC > See that most food stores are on the perimeter for this NTA, with only a handful in the interior roads.
# MAGIC 
# MAGIC _Again, screenshot included for reading outside Databricks and to show some additional rendering tweaks._

# COMMAND ----------

plt.imshow(img.imread("Resources/store_street_kring_20.png"), aspect='auto')
plt.show()

# COMMAND ----------

county = "Bronx"
ntacodes = ['BX22']

pdf_store = spark.table("store").select("geom_wkt", "entity_name", "county", "establishment_type", "license_number",).filter(f"county == '{county}'").toPandas()
pdf_store_cell = df_store_cell.select("cell_str", "zip_code", "county").filter(col("cell_str").isNotNull()).filter(f"county == '{county}'").toPandas()
pdf_nta_geom = spark.table("nta_pop").select("geom_wkt", "county", "ntaname", "pop_year_2010", "pop_year_2000", "ntacode").filter(col("ntacode").isin(*ntacodes)).toPandas()

pdf_street_no_k20_cell = df_k20_street.select("street_cell_str", "ntacode", "ntaname", "is_street_cell", "is_k20").filter("is_street_cell == True").filter("is_k20 == False").filter(col("ntacode").isin(*ntacodes)).toPandas()
pdf_street_k20_cell = df_k20_street.select("street_cell_str", "ntacode", "ntaname", "is_street_cell", "is_k20").filter("is_street_cell == True").filter("is_k20 == True").filter(col("ntacode").isin(*ntacodes)).toPandas()

# -- Optional: Layer Tweaks --
# - For `street_no_k20_cell` set color to red
# - turn off fill for `nta_pop` geom_wkt
map_street_nta_config = {
  'version': 'v1', 
  'mapState': {
    'latitude': 40.90, 
    'longitude': -73.908, 
    'zoom': 13.5
  }, 
  'mapStyle': {'styleType': 'satellite'}
} 

display_kepler(
  KeplerGl(
    config=map_street_nta_config,
    data={
      'store': pdf_store,
      'store_cell': pdf_store_cell,
      'street_no_k20_cell': pdf_street_no_k20_cell,
      'street_k20_cell': pdf_street_k20_cell,
      'nta_pop': pdf_nta_geom
    },
    show_docs=False
  )
)
