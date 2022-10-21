# Databricks notebook source
# MAGIC %md # NYC Census Tracts, Blocks & NTAs
# MAGIC 
# MAGIC > 2020 Census Tracts and Blocks from the US Census for New York City. These boundary files are derived from the US Census Bureau's TIGER data products and have been geographically modified to fit the New York City base map. [[2020 Census Blocks Tabular](https://data.cityofnewyork.us/City-Government/2020-Census-Blocks-Tabular/wmsu-5muw)]
# MAGIC 
# MAGIC __Tables Generated__
# MAGIC <p/>
# MAGIC 
# MAGIC 1. `census_tract`
# MAGIC 1. `nta_pop`
# MAGIC 1. `census_block`
# MAGIC 
# MAGIC * CSV from https://data.cityofnewyork.us/api/views/wmsu-5muw/rows.csv?accessType=DOWNLOAD
# MAGIC * Alt: GeoJSON from https://data.cityofnewyork.us/api/geospatial/wmsu-5muw?method=export&format=GeoJSON
# MAGIC 
# MAGIC --- 
# MAGIC __Author: Michael Johns <mjohns@databricks.com>__  
# MAGIC _Last Modified: 17 OCT 2022_

# COMMAND ----------

# MAGIC %md
# MAGIC __More on [Census Data](https://learn.arcgis.com/en/related-concepts/united-states-census-geography.htm):__
# MAGIC 
# MAGIC > (1) Census tracts are statistical subdivisions of a county that aim to have roughly 4,000 inhabitants. Tract boundaries are usually visible features, such as roads or rivers, but they can also follow the boundaries of national parks, military reservations, or American Indian reservations. Tracts are designed to be fairly homogeneous with respect to demographic and economic conditions when they are first established. When a census tract experiences growth and the internal population grows beyond 8,000 persons, the tract is split up. This review and revision process is conducted every decade with collaboration from local planning agencies.
# MAGIC 
# MAGIC > (2) A block group is a subdivision of a census tract and contains a cluster of blocks. Block groups usually have between 250 and 550 housing units.
# MAGIC 
# MAGIC > (3) A census block is the smallest geographic census unit. Blocks can be bounded by visible features—such as streets—or by invisible boundaries, such as city limits. Census blocks are often the same as ordinary city blocks. Census blocks change every decade.
# MAGIC 
# MAGIC <img src="https://learn.arcgis.com/en/related-concepts/GUID-D7AA4FD1-E7FE-49D7-9D11-07915C9ACC68-web.png" width="35%"/>

# COMMAND ----------

# MAGIC %md __More on Neighborhood Tabulation Areas (NTA)__
# MAGIC 
# MAGIC > NTAs are aggregations of census tracts that are subsets of New York City's 55 Public Use Micro data Areas (PUMAs) -- from [NYC Data](https://www1.nyc.gov/site/planning/data-maps/open-data/census-download-metadata.page)
# MAGIC 
# MAGIC <img src="https://www1.nyc.gov/assets/planning/images/content/pages/planning-level/nyc-population/new-population/dcp-atlas-nyc-2020-census-geogs.jpg" width="25%"/>

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
# MAGIC 4. To use [PyShp](https://github.com/GeospatialPython/pyshp) OSS library for pure python shapefile reading (and writing, not shown):
# MAGIC   * Install with `%pip install pyshp`
# MAGIC   * Import with `import shapefile`
# MAGIC 
# MAGIC __Note: If you hit `H3_NOT_ENABLED` [[docs](https://docs.databricks.com/error-messages/h3-not-enabled-error-class.html#h3_not_enabled-error-class)]__
# MAGIC 
# MAGIC > `h3Expression` is disabled or unsupported. Consider enabling Photon or switch to a tier that supports H3 expressions. [[AWS](https://www.databricks.com/product/aws-pricing) | [Azure](https://azure.microsoft.com/en-us/pricing/details/databricks/) | [GCP](https://www.databricks.com/product/gcp-pricing)]
# MAGIC 
# MAGIC __Recommend running on DBR 11.3+ for maximizing photonized functions.__

# COMMAND ----------

# MAGIC %pip install databricks-mosaic pyshp --quiet

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

# MAGIC %md ## Load NYC Census Tracts (Shapefile)
# MAGIC 
# MAGIC > Will use pyshp to read the shapefile
# MAGIC 
# MAGIC * Data from https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyct2020_22b.zip

# COMMAND ----------

#%sh 
#wget -O nyc_census_tracts.shapefile.zip "https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyct2020_22b.zip"
#mkdir -p $etl_dir_fuse
#rm $etl_dir_fuse/nyc_census_tracts.shapefile.zip
#cp nyc_census_tracts.shapefile.zip $etl_dir_fuse/

# COMMAND ----------

def _shapefile_reader(shapefile_path:str, java_friendly:bool=True):
  """
  Read shapefile (focused on zip file)
  - flattens properties and geometry as geojson
  - java_friendly will handle converting:
    1. from single quotes to double
    2. from tuples to lists
  """
  import shapefile
  
  with shapefile.Reader(shapefile_path) as shp:
      shape_records = []

      # Iterate through each shape record
      for shape in shp.shapeRecords():
        shape_record = shape.record.as_dict() # Read record
        geojson = {'geojson':shape.shape.__geo_interface__.__str__()} # Read shapefile GeoJSON
        if java_friendly:
          geojson['geojson']  = geojson['geojson'].replace("'",'"').replace("(","[").replace(")","]")
        shape_records.append({**shape_record, **geojson}) # Concatenate and append
  
  return(shape_records)


@udf(returnType=ArrayType(MapType(StringType(), StringType())))
def shapefile_reader(shapefile_path:str, java_friendly:bool=True):
  return _shapefile_reader(shapefile_path, java_friendly=java_friendly)

spark.udf.register("shapefile_reader", shapefile_reader) # <-- register for SQL

# COMMAND ----------

# MAGIC %md ### Note: CRS is 2263 for the shapefile data
# MAGIC 
# MAGIC > We can transform with Mosaic to 4326
# MAGIC 
# MAGIC __Shapefile Reading__
# MAGIC 
# MAGIC 1. Custom function that uses `pyshp` library to read properties and geometry (as geojson)
# MAGIC 1. Only 1 shapefile, so directly calling `_shapefile_reader` to create Spark DataFrame
# MAGIC 1. Could use in UDF for _n_ shapefiles loaded in parrallel
# MAGIC 
# MAGIC __Mosaic Functions__
# MAGIC 
# MAGIC ```
# MAGIC .withColumn("geom_2263", mos.st_geomfromgeojson("geojson"))
# MAGIC .withColumn("geom_2263", mos.st_setsrid("geom_2263", F.lit(2263)))
# MAGIC .withColumn("geom", mos.st_transform("geom_2263", F.lit(4326)))
# MAGIC .withColumn("geom_wkt", mos.st_astext("geom"))
# MAGIC .withColumn("is_valid", mos.st_isvalid("geom"))
# MAGIC .withColumn("is_coords_4326", mos.st_hasvalidcoordinates("geom", F.lit("EPSG:4326"), F.lit('bounds')))
# MAGIC .withColumn("was_coords_2263", mos.st_hasvalidcoordinates("geom_2263", F.lit("EPSG:2263"), F.lit('reprojected_bounds')))
# MAGIC ```

# COMMAND ----------

_df_census_tract = (
  spark
    .createDataFrame(
      _shapefile_reader(f"{etl_dir_fuse}/nyc_census_tracts.shapefile.zip")
    )
    .withColumn("geom_2263", mos.st_geomfromgeojson("geojson"))
    .withColumn("geom_2263", mos.st_setsrid("geom_2263", F.lit(2263)))
    .withColumn("geom", mos.st_transform("geom_2263", F.lit(4326)))
    .withColumn("geom_wkt", mos.st_astext("geom"))
    .withColumn("is_valid", mos.st_isvalid("geom"))
    .withColumn("is_coords_4326", mos.st_hasvalidcoordinates("geom", F.lit("EPSG:4326"), F.lit('bounds')))
    .withColumn("was_coords_2263", mos.st_hasvalidcoordinates("geom_2263", F.lit("EPSG:2263"), F.lit('reprojected_bounds')))
      .selectExpr(
        "* except(geojson, geom_wkt, geom, is_valid, is_coords_4326, was_coords_2263, geom_2263)",
        "geom_wkt", "is_valid", "is_coords_4326", "geom", "was_coords_2263"
      )
)

_df_census_tract = _df_census_tract.toDF(*[c.lower().replace(' ','_') for c in _df_census_tract.columns])

# _df_census_tract.display()

# COMMAND ----------

# MAGIC %md ### Write to Delta
# MAGIC 
# MAGIC > Results in `census_tract` table

# COMMAND ----------

df_census_tract = write_to_delta(_df_census_tract, f"{etl_dir}/census_tract", "census_tract", db_name, zorder_col='ct2020', register_table=True, overwrite=False)
print(f"count? {df_census_tract.count():,}")
df_census_tract.display()

# COMMAND ----------

df_census_tract_kepler = df_census_tract.drop("geom")

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df_census_tract_kepler "geom_wkt" "geometry" 5_000

# COMMAND ----------

# MAGIC %md ## Neighborhood Tabulation Area (GeoJSON)
# MAGIC 
# MAGIC > NTAs are aggregations of census tracts that are subsets of New York City's 55 Public Use Microdata Areas (PUMAs) [[1](https://www1.nyc.gov/site/planning/data-maps/open-data/census-download-metadata.page)]
# MAGIC 
# MAGIC * GeoJSON -- https://data.cityofnewyork.us/api/geospatial/d3qk-pfyz?method=export&format=GeoJSON
# MAGIC * Demographics
# MAGIC   * [New York City Population By Neighborhood Tabulation Areas](https://data.cityofnewyork.us/City-Government/New-York-City-Population-By-Neighborhood-Tabulatio/swpk-hqdp) -- This report shows change in population from 2000 to 2010 for each NTA.
# MAGIC   * [Demographics and profiles at the Neighborhood Tabulation Area (NTA) level](https://data.cityofnewyork.us/City-Government/Demographics-and-profiles-at-the-Neighborhood-Tabu/hyuz-tij8) -- Table of ACS Demographics and profile represented at the NTA level. __This is human-readable format only.__
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

#%sh 
#wget -O nyc_nta.geojson "https://data.cityofnewyork.us/api/geospatial/d3qk-pfyz?method=export&format=GeoJSON"
#mkdir -p $etl_dir_fuse
#rm $etl_dir_fuse/nyc_nta.geojson
#cp nyc_nta.geojson $etl_dir_fuse/

# COMMAND ----------

_df_nta = (
  spark.read
    .option("multiline", "true")
    .format("json")
    .load(f"{etl_dir}/nyc_nta.geojson")
      .select("type", F.explode(col("features")).alias("feature"))
      .select("type", col("feature.properties").alias("properties"), F.to_json(col("feature.geometry")).alias("json_geometry"))
    .withColumn("geom", mos.st_geomfromgeojson("json_geometry"))
    .withColumn("geom", mos.st_setsrid("geom", F.lit(4326)))
    .withColumn("geom_wkt", mos.st_astext("geom"))
    .withColumn("is_valid", mos.st_isvalid("geom"))
    .withColumn("is_coords_4326", mos.st_hasvalidcoordinates("geom", F.lit("EPSG:4326"), F.lit('bounds')))
      .select("properties.*", "geom_wkt", "is_valid", "is_coords_4326", "geom")
    .withColumn(
      "county", 
      F.when(col("borocode") == 1, "New York")
      .when(col("borocode") == 2, "Bronx")
      .when(col("borocode") == 3, "Kings")
      .when(col("borocode") == 4, "Queens")
      .when(col("borocode") == 5, "Richmond")
      .otherwise("UKN")
    )  
)
_df_nta.display()

# COMMAND ----------

# MAGIC %md ### Population Data (CSV)
# MAGIC 
# MAGIC > New York City Population By Neighborhood Tabulation Areas

# COMMAND ----------

#%sh 
#wget -O nyc_nta_population.csv "https://data.cityofnewyork.us/api/views/swpk-hqdp/rows.csv?accessType=DOWNLOAD"
#mkdir -p $etl_dir_fuse
#rm -f $etl_dir_fuse/nyc_nta_population.csv
#cp nyc_nta_population.csv $etl_dir_fuse/

# COMMAND ----------

_df_pop = (
  spark
    .read
      .csv(f"{etl_dir}/nyc_nta_population.csv", header=True, inferSchema=True)
)

_df_pop = _df_pop.toDF(*[c.lower().replace(' ','_') for c in _df_pop.columns])

_df_pop = (
  _df_pop
    .selectExpr("nta_code as ntacode", "year", "population")
      .groupBy("ntacode")
      .pivot("year").sum("population")
    .selectExpr("ntacode", "`2000` as pop_year_2000", "`2010` as pop_year_2010")
)

display(_df_pop)

# COMMAND ----------

# MAGIC %md ### Join Population

# COMMAND ----------

_df_nta_pop = (
  _df_nta
    .join(
      _df_pop, 
      ["ntacode"]
    )
    .selectExpr("ntacode","ntaname","pop_year_2000","pop_year_2010", "* except(ntacode, ntaname, pop_year_2000, pop_year_2010)")
) 
print(f"nta-pop count? {_df_nta_pop.count():,}")
print(f"...nta count? {_df_nta.count():,}")
print(f"...pop count? {_df_pop.count():,}")
# display(_df_nta_pop)

# COMMAND ----------

# MAGIC %md ### Write to Delta
# MAGIC 
# MAGIC > Results in `nta_pop` table

# COMMAND ----------

df_nta_pop = write_to_delta(_df_nta_pop, f"{etl_dir}/nta_pop", "nta_pop", db_name, zorder_col='ntacode', register_table=True, overwrite=False)
print(f"count? {df_nta_pop.count():,}")
df_nta_pop.display()

# COMMAND ----------

df_nta_pop_kepler = df_nta_pop.drop("geom")

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df_nta_pop_kepler "geom_wkt" "geometry"

# COMMAND ----------

# MAGIC %md ## Load NYC Census Blocks (CSV)
# MAGIC 
# MAGIC > Will use built-in CSV reader
# MAGIC 
# MAGIC __Mosaic Functions__
# MAGIC 
# MAGIC ```
# MAGIC .withColumn("geom", mos.st_geomfromwkt("the_geom"))
# MAGIC .withColumn("geom", mos.st_setsrid("geom", F.lit(4326)))
# MAGIC .withColumn("is_valid", mos.st_isvalid("geom"))
# MAGIC .withColumn("is_coords_4326", mos.st_hasvalidcoordinates("geom", F.lit("EPSG:4326"), F.lit('bounds')))
# MAGIC ```

# COMMAND ----------

#%sh 
#wget -O nyc_census_blocks.csv "https://data.cityofnewyork.us/api/views/wmsu-5muw/rows.csv?accessType=DOWNLOAD"
#mkdir -p $etl_dir_fuse
#cp nyc_census_blocks.csv $etl_dir_fuse/

# COMMAND ----------

_df_census_block = (
  spark
    .read
    .csv(f"{etl_dir}/nyc_census_blocks.csv", header=True, inferSchema=True)
    .withColumn("geom", mos.st_geomfromwkt("the_geom"))
    .withColumn("geom", mos.st_setsrid("geom", F.lit(4326)))
    .withColumn("is_valid", mos.st_isvalid("geom"))
    .withColumn("is_coords_4326", mos.st_hasvalidcoordinates("geom", F.lit("EPSG:4326"), F.lit('bounds')))
    .withColumn(
      "county", 
      F.when(col("boroname") == "Manhattan",     "New York")
      .when(col("boroname")  == "Bronx",         "Bronx")
      .when(col("boroname")  == "Brooklyn",      "Kings")
      .when(col("boroname")  == "Queens",        "Queens")
      .when(col("boroname")  == "Staten Island", "Richmond")
      .otherwise("UKN")
    )  
)

_df_census_block = _df_census_block.toDF(*[c.lower().replace(' ','_') for c in _df_census_block.columns])

# display(_df_census_block)

# COMMAND ----------

# MAGIC %md ### Write to Delta
# MAGIC 
# MAGIC > Results in `census_block` table

# COMMAND ----------

df_census_block = write_to_delta(_df_census_block, f"{etl_dir}/census_block", "census_block", db_name, zorder_col='BoroName', register_table=True, overwrite=False)
print(f"count? {df_census_block.count():,}")
df_census_block.display()

# COMMAND ----------

df_census_block_kepler = df_census_block.select("the_geom", "boroname", "cb2020").filter("boroname == 'Manhattan'")

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df_census_block_kepler "the_geom" "geometry" 50_000

# COMMAND ----------

# %sql show tables
