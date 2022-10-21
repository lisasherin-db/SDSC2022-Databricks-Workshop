# Databricks notebook source
# MAGIC %pip install databricks-mosaic

# COMMAND ----------

from pyspark.sql import functions as F
import mosaic as mos

from mosaic import enable_mosaic
enable_mosaic(spark, dbutils)

# COMMAND ----------

user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

raw_path = f"dbfs:/tmp/mosaic/{user_name}/data/sdsc"
dbutils.fs.mkdirs(raw_path)

print(f"The raw data will be stored in {raw_path}")

# COMMAND ----------

dbutils.fs.ls(raw_path)

# COMMAND ----------

def read_geojson(path):
  df = (spark.read
    .option("multiline", "true")
    .format("json")
    .load(path)
    .select("type", F.explode(F.col("features")).alias("feature"))
    .select("type", F.col("feature.properties").alias("properties"), F.col("feature.geometry").alias("json_geometry")))
    #.withColumn("geometry", mos.st_aswkt(mos.st_geomfromgeojson("json_geometry"))))
  return df

# COMMAND ----------

buildings_df = read_geojson("dbfs:/tmp/mosaic/milos.colic@databricks.com/data/sdsc/nyc_building_footprints.geojson")
school_districts_df = read_geojson("dbfs:/tmp/mosaic/milos.colic@databricks.com/data/sdsc/nyc_school_districts.geojson")
subway_stations_df = read_geojson("dbfs:/tmp/mosaic/milos.colic@databricks.com/data/sdsc/nyc_schools_in_construction.geojson")

# COMMAND ----------

buildings_df = buildings_df.withColumn(
  "x", F.expr("json_geometry.coordinates[0]").cast("double")
).withColumn(
  "y", F.expr("json_geometry.coordinates[1]").cast("double")
).withColumn(
  "geometry", mos.st_point("x", "y")
).drop("json_geometry", "x", "y")

# COMMAND ----------

buildings_df.write.format("delta").saveAsTable("sdsc_buildings")

# COMMAND ----------

buildings_df = spark.read.table("sdsc_buildings").repartition(200).withColumn("geometry", mos.st_aswkt("geometry"))

# COMMAND ----------

buildings_df.display()

# COMMAND ----------

buildings_df = buildings_df.repartition(200)
buildings_df.write.format("delta").saveAsTable("temp_buildings")

# COMMAND ----------

buildings_df = spark.read.table("temp_buildings")

# COMMAND ----------

tripsTable = spark.table("delta.`/databricks-datasets/nyctaxi/tables/nyctaxi_yellow`").withColumn(
  "pickup_point", mos.st_aswkt(mos.st_point("pickup_longitude", "pickup_latitude"))
).withColumn(
  "dropoff_point", mos.st_aswkt(mos.st_point("dropoff_longitude", "dropoff_latitude"))
)

# COMMAND ----------



# COMMAND ----------

tripsTable.display()

# COMMAND ----------

tripsTable = tripsTable.withColumn("pickup_h3", mos.grid_pointascellid("pickup_point", F.lit(8))).withColumn("dropoff_h3", mos.grid_pointascellid("dropoff_point", F.lit(8)))
buildings_df = buildings_df.withColumn("building_h3", mos.grid_pointascellid("geometry", F.lit(8)))

# COMMAND ----------



# COMMAND ----------

pickups = tripsTable.orderBy(F.rand()).limit(100000).select("pickup_h3", "pickup_point")

# COMMAND ----------

pickups.write.format("delta").saveAsTable("sdsc_trips")

# COMMAND ----------

pickups.display()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC pickups "pickup_h3" "h3" 100

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC buildings_df "building_h3" "h3" 100

# COMMAND ----------

def knn_ring(geom, n):
  return mos.grid_tessellateexplode(mos.st_buffer(geom, n*F.lit(0.01)), F.lit(8)).alias(f"ring_{n}")

# COMMAND ----------

spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")

# COMMAND ----------

buildings_df = buildings_df.repartition(200)

# COMMAND ----------

buildings_df = spark.read.table("temp_buildings")

# COMMAND ----------

buildings_df = spark.read.table("temp_buildings")
buildings_df = buildings_df.repartition(200).limit(500).withColumn(
  "building_h3", mos.grid_pointascellid("geometry", F.lit(8))
).where(
  "geometry is not null"
).withColumn(
  "knn_ring_1", knn_ring("geometry", 1)
).where(
  "knn_ring_1 is not null"
).select(
  "type",
  "geometry",
  "building_h3",
  "knn_ring_1"
)

# COMMAND ----------

buildings_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("knn_ring_1")

# COMMAND ----------

buildings_df = spark.read.table("knn_ring_1").select("building_h3", "geometry", "knn_ring_1.index_id").withColumn("buffer", mos.st_buffer("geometry", F.lit(0.01)))

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC buildings_df "index_id" "h3" 1000

# COMMAND ----------

buildings_df = spark.read.table("temp_buildings")
buildings_df = buildings_df.repartition(200).limit(500).withColumn(
  "building_h3", mos.grid_pointascellid("geometry", F.lit(8))
).where(
  "geometry is not null"
).withColumn(
  "knn_ring_1", knn_ring("geometry", 1)
).withColumn(
  "knn_ring_2", knn_ring("geometry", 2)
).where(
  "knn_ring_1 is not null"
).select(
  "type",
  "geometry",
  "building_h3",
  "knn_ring_1",
  "knn_ring_2"
).where("knn_ring_1.index_id != knn_ring_2.index_id")

# COMMAND ----------



# COMMAND ----------

buildings_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("knn_ring_2")

# COMMAND ----------

buildings_df = spark.read.table("knn_ring_2").where("not knn_ring_2.is_core").select("building_h3", "geometry", "knn_ring_2.index_id").withColumn("buffer", mos.st_buffer("geometry", F.lit(0.02)))

# COMMAND ----------

buildings_df.display()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC buildings_df "index_id" "h3" 5000

# COMMAND ----------

buildings_df = spark.read.table("temp_buildings")
buildings_df = buildings_df.repartition(200).limit(500).withColumn(
  "building_h3", mos.grid_pointascellid("geometry", F.lit(8))
).where(
  "geometry is not null"
).withColumn(
  "knn_ring_3", knn_ring("geometry", 3)
).select(
  "type",
  "geometry",
  "building_h3",
  "knn_ring_3"
)

# COMMAND ----------

buildings_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("knn_ring_3")

# COMMAND ----------

buildings_df = spark.read.table("knn_ring_3").where("not knn_ring_3.is_core").select("building_h3", "geometry", "knn_ring_3.index_id").withColumn("buffer", mos.st_buffer("geometry", F.lit(0.03)))

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC buildings_df "index_id" "h3" 500

# COMMAND ----------

buildings_df = spark.read.table("temp_buildings")
buildings_df = buildings_df.repartition(200).limit(500).withColumn(
  "building_h3", mos.grid_pointascellid("geometry", F.lit(8))
).where(
  "geometry is not null"
).withColumn(
  "knn_ring_4", knn_ring("geometry", 4)
).select(
  "type",
  "geometry",
  "building_h3",
  "knn_ring_4"
)

# COMMAND ----------

buildings_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("knn_ring_4")

# COMMAND ----------

buildings_df = spark.read.table("knn_ring_4").where("not knn_ring_4.is_core").select("building_h3", "geometry", "knn_ring_4.index_id").withColumn("buffer", mos.st_buffer("geometry", F.lit(0.04)))

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC buildings_df "index_id" "h3" 500

# COMMAND ----------

buildings_df = spark.read.table("temp_buildings").limit(100).cache()
pickups_df = spark.read.table("sdsc_trips").select("pickup_point", "pickup_h3").limit(10000).cache()

# COMMAND ----------

from mosaic.utils.kepler_config import mosaic_kepler_config
from keplergl import KeplerGl

m1 = KeplerGl(config=mosaic_kepler_config)

# COMMAND ----------

m1.add_data(buildings_df.select("geometry").toPandas(), name = "buildings")
m1.add_data(pickups_df.toPandas(), name = "pickups")

# COMMAND ----------

displayHTML(m1._repr_html_()
            .decode("utf-8").replace(".height||400", f".height||{800}")
            .replace(".width||400", f".width||{800}"))

# COMMAND ----------

def build_iteration(df, n):
  df = (df.repartition(200)
          .where("geometry is not null")
          .withColumn(
            "point_h3", 
            mos.grid_pointascellid("geometry", F.lit(8))
          ).withColumn(
            "knn_ring", knn_ring("geometry", n)
          ).select(
            "type",
            "geometry",
            "point_h3",
            "knn_ring"
          ))
  df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"knn_ring_{n}")

# COMMAND ----------

for i in range(0, 10):
  build_iteration(buildings_df.limit(20), i)

# COMMAND ----------

def plot_iteration(n, f):
  m1 = KeplerGl(config=mosaic_kepler_config)
  buildings_df = spark.read.table(f"knn_ring_{n}").withColumn(
                    "index_id", F.lower(F.conv(F.col("knn_ring.index_id"), 10, 16))
                ).where(f"{f} or (not knn_ring.is_core)")
  
  m1.add_data(pickups_df.toPandas(), name = "pickups")
  m1.add_data(buildings_df.select("index_id", "geometry").toPandas(), name = "buildings")
  
  displayHTML(m1._repr_html_()
            .decode("utf-8").replace(".height||400", f".height||{800}")
            .replace(".width||400", f".width||{800}"))

# COMMAND ----------

plot_iteration(1, True)

# COMMAND ----------

plot_iteration(3, False)

# COMMAND ----------

plot_iteration(5, False)

# COMMAND ----------

plot_iteration(7, False)

# COMMAND ----------

import re

import h3
import pandas as pd
from IPython.core.magic import Magics, cell_magic, magics_class
from keplergl import KeplerGl
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, conv, lower, lit

from mosaic.api.accessors import st_astext, st_aswkt
from mosaic.api.constructors import st_geomfromwkt, st_geomfromwkb
from mosaic.api.functions import st_centroid2D, grid_pointascellid, grid_boundaryaswkb, st_setsrid, st_transform
from mosaic.config import config
from mosaic.utils.kepler_config import mosaic_kepler_config


@magics_class
class MosaicKepler2(Magics):

    """
    A magic command for visualizing data in KeplerGl.
    """

    def __init__(self, shell):
      Magics.__init__(self, shell)
      self.bng_crsid = 27700
      self.osgb36_crsid = 27700
      self.wgs84_crsid = 4326

    @staticmethod
    def displayKepler(map_instance, height, width):

        """
        Display Kepler map instance in Jupyter notebook.

        Parameters:
        ------------
        map_instance: KeplerGl
            Kepler map instance
        height: int
            Height of the map
        width: int
            Width of the map

        Returns:
        -----------
        None

        Example:
            displayKepler(map_instance, 600, 800)
        """
        decoded = (
            map_instance._repr_html_()
            .decode("utf-8")
            .replace(".height||400", f".height||{height}")
            .replace(".width||400", f".width||{width}")
        )
        ga_script_redacted = re.sub(
            r"\<script\>\(function\(i,s,o,g,r,a,m\).*?GoogleAnalyticsObject.*?(\<\/script\>)",
            "",
            decoded,
            flags=re.DOTALL,
        )
        async_script_redacted = re.sub(
            r"s\.a\.createElement\(\"script\",\{async.*?\}\),",
            "",
            ga_script_redacted,
            flags=re.DOTALL,
        )
        config.notebook_utils.displayHTML(async_script_redacted)

    @staticmethod
    def get_spark_df(table_name):
        """
        Constructs a Spark DataFrame from table name, Pandas DataFrame or Spark DataFrame.

        Parameters:
        ------------
        table_name: str
            Name of the table

        Returns:
        -----------
        Spark DataFrame

        Example:
            get_spark_df(table_name)
        """

        try:
            table = config.ipython_hook.ev(table_name)
            if isinstance(table, pd.DataFrame):  # pandas dataframe
                data = config.mosaic_spark.createDataFrame(table)
            elif isinstance(table, DataFrame):  # spark dataframe
                data = table
            else:  # table name
                data = config.mosaic_spark.read.table(table)
        except NameError:
            try:
                data = config.mosaic_spark.read.table(table_name)
            except:
                raise Exception(f"Table name reference invalid.")
        return data

    @staticmethod
    def set_centroid(pandas_data, feature_type, feature_name):

        """
        Sets the centroid of the geometry column.

        Parameters:
        ------------
        pandas_data: Pandas DataFrame
            Pandas DataFrame containing the geometry column to be visualized in KeplerGl.
        feature_type: str
            Type of the feature column to be visualized in KeplerGl.
            This can be "h3", "bng" or "geometry".
            "geometry" represents geometry column with CRSID 4326.
            "geometry(bng)" or "geometry(osgb36)" represents geometry column with CRSID 27700.
            "geometry(23456)" represents geometry column with 23456 where 23456 is the EPSG code.
        feature_name: str
            Name of the column containing the geometry to be visualized in KeplerGl.

        Returns:
        -----------
        None

        Example:
            set_centroid(pdf, "h3", "hex_id")
            set_centroid(pdf, "bng", "bng_id")
            set_centroid(pdf, "geometry", "geom")
            set_centroid(pdf, "geometry(bng)", "geom")
            set_centroid(pdf, "geometry(osgb36)", "geom")
            set_centroid(pdf, "geometry(27700)", "geom")
            set_centroid(pdf, "geometry(23456)", "geom")
        """

        tmp_sdf = config.mosaic_spark.createDataFrame(pandas_data.iloc[:1])

        if feature_type == "h3":
            tmp_sdf = tmp_sdf.withColumn(feature_name, grid_boundaryaswkb(feature_name))

        centroid = (
            tmp_sdf.select(st_centroid2D(feature_name))
            .limit(1)
            .collect()[0][0]
        )

        # set to centroid of a geom
        mosaic_kepler_config["config"]["mapState"]["latitude"] = centroid[1]
        mosaic_kepler_config["config"]["mapState"]["longitude"] = centroid[0]

    @cell_magic
    def mosaic_kepler2(self, *args):

        """
        A magic command for visualizing data in KeplerGl.

        Parameters:
        ------------
        args: str
            Arguments passed to the magic command.
            The first argument is the name of the table to be visualized in KeplerGl.
            The second argument is the type of the feature column to be visualized in KeplerGl.
            This can be "h3", "bng" or "geometry".
            "geometry" represents geometry column with CRSID 4326.
            "geometry(bng)" or "geometry(osgb36)" represents geometry column with CRSID 27700.
            "geometry(23456)" represents geometry column with 23456 where 23456 is the EPSG code.

        Returns:
        -----------
        None

        Example:
            %mosaic_kepler table_name geometry_column h3 [limit]
            %mosaic_kepler table_name geometry_column bng [limit]
            %mosaic_kepler table_name geometry_column geometry [limit]
            %mosaic_kepler table_name geometry_column geometry(bng) [limit]
            %mosaic_kepler table_name geometry_column geometry(osgb36) [limit]
            %mosaic_kepler table_name geometry_column geometry(27700) [limit]
            %mosaic_kepler table_name geometry_column geometry(23456) [limit]
        """

        inputs = [
            i
            for i in " ".join(list(args)).replace("\n", " ").replace('"', "").split(" ")
            if len(i) > 0
        ]

        if len(inputs) != 3 and len(inputs) != 4:
            raise Exception(
                "Mosaic Kepler magic requires table name, feature column and feature type all to be provided. "
                + "Limit is optional (default 1000)."
            )

        table_name = inputs[0]
        feature_name = inputs[1]
        feature_type = inputs[2]
        limit_ctn = 1000
        if len(inputs) == 4:
            limit_ctn = int(inputs[3])
        data = self.get_spark_df(table_name)
        feature_col_dt = [dt for dt in data.dtypes if dt[0] == feature_name][0]

        if feature_type == "h3":
            if feature_col_dt[1] == "bigint":
                data = data.withColumn(
                    feature_name, lower(conv(col(feature_name), 10, 16))
                )
        elif feature_type == "bng":
            data = (data
                .withColumn(feature_name, grid_boundaryaswkb(feature_name))
                .withColumn(feature_name, st_geomfromwkb(feature_name))
                .withColumn(
                    feature_name,
                    st_transform(st_setsrid(feature_name, lit(self.bng_crsid)), lit(self.wgs84_crsid))
                )
                .withColumn(feature_name, st_aswkt(feature_name)))
        elif feature_type == "geometry":
            data = data.withColumn(feature_name, st_astext(col(feature_name)))
        elif re.search("^geometry\(.*\)$", feature_type).start() != None:
            crsid = feature_type.replace("geometry(", "").replace(")", "").lower()
            if crsid == "bng" or crsid == "osgb36":
                crsid = self.bng_crsid
            else:
                crsid = int(crsid)
            data = (data
                .withColumn(feature_name, st_geomfromwkt(st_aswkt(feature_name)))
                .withColumn(
                    feature_name,
                    st_transform(st_setsrid(feature_name, lit(crsid)), lit(self.wgs84_crsid))
                )
                .withColumn(feature_name, st_aswkt(feature_name)))
        else:
            raise Exception(f"Unsupported geometry type: {feature_type}.")

        allowed_schema = [
            field.name
            for field in data.schema.fields
            if field.dataType.typeName() in ["string", "long", "integer", "double"]
        ]
        data = data.select(*allowed_schema)
        pandas_data = data.limit(limit_ctn).toPandas()

        self.set_centroid(pandas_data, feature_type, feature_name)

        m1 = KeplerGl(config=mosaic_kepler_config)
        m1.add_data(data=pandas_data, name=table_name)

        self.displayKepler(m1, 800, 1200)


# COMMAND ----------

get_ipython().register_magics(MosaicKepler2)

# COMMAND ----------

buildings_df = spark.read.table("knn_ring_4").where("not knn_ring.is_core").select("point_h3", "geometry", "knn_ring.index_id").withColumn("buffer", mos.st_buffer("geometry", lit(0.04)))

# COMMAND ----------

# MAGIC %%mosaic_kepler2
# MAGIC buildings_df "point_h3" "h3" 50

# COMMAND ----------

df = spark.read.table("checkpoint_table_matches")

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.where("neighbour_number < 200")

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df "pickup_point" "geometry" 10000

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC df "index_id" "h3" 10000

# COMMAND ----------


