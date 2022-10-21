# Databricks notebook source
def is_dbfs_dir_exists(dbfs_dir, debug=True):
  """
  Convienence to verify if a dbfs dir exists.
  Returns True | False
  """
  exists = False
  try:
    if len(dbutils.fs.ls(dbfs_dir)) > 0:
      exists = True
  except Exception:
    print(f"...'{dbfs_dir}' does not exist.") if debug==True else None
    pass
  return exists

# COMMAND ----------

def handle_delta_optimize_register(delta_dir, tbl_name, db_name, zorder_col=None, register_table=True):
  """
  """
  # -- Optimze + Z-Order
  if zorder_col is not None:
    sql(f"""OPTIMIZE delta.`{delta_dir}` ZORDER BY ({zorder_col})""")
  
  # -- Register Table
  if register_table:
    sql(f"""DROP TABLE IF EXISTS {db_name}.{tbl_name}""") # <-- register table with metastore
    sql(f"""CREATE TABLE {db_name}.{tbl_name} 
                USING DELTA
                LOCATION '{delta_dir}'
        """)
    return spark.table(f"{db_name}.{tbl_name}")
  else:
    return spark.read.load(delta_dir)

# COMMAND ----------

def write_to_delta(_df, delta_dir, tbl_name, db_name, zorder_col=None, register_table=True, overwrite=False, debug=True):
  """
  - (1) write dataframe to delta lake
  - (2) optionally: optimize + z-order
  - (3) register table
  This uses various existing variables as well.
  """
  exists = is_dbfs_dir_exists(delta_dir)
  if exists and not overwrite:
    print(f"...returning, '{delta_dir}' exists and is not empty and overwrite=False") if debug==True else None
    if register_table:
      return spark.table(f"{db_name}.{tbl_name}")
    else:
      return spark.read.load(delta_dir)
  
  # -- Write to Delta Lake
  dbutils.fs.rm(delta_dir, True)
  (
    _df
      .write
        .mode("overwrite")
        .option("mergeSchema", "true")
      .save(delta_dir)
  )
  
  return handle_delta_optimize_register(delta_dir, tbl_name, db_name, zorder_col=zorder_col, register_table=register_table)

# COMMAND ----------

# -- DOING X,Y from WKT VIA MOSAIC 
# - This is an alternative
# @udf(returnType=FloatType())
# def lon_from_point(wkt:str) -> float:
#   """
#   Get the 'x' value
#   """
#   if wkt is None:
#     return None
#   shp = shapely.wkt.loads(wkt)
#   if shp.type == 'Point':
#     return shp.coords[0][0]
#   return None

# @udf(returnType=FloatType())
# def lat_from_point(wkt:str) -> float:
#   """
#   Get the 'y' value
#   """
#   if wkt is None:
#     return None
#   shp = shapely.wkt.loads(wkt)
#   if shp.type == 'Point':
#     return shp.coords[0][1]
#   return None

# spark.udf.register("lon_from_point", lon_from_point) # <-- register for SQL
# spark.udf.register("lat_from_point", lat_from_point) # <-- register for SQL
