# Databricks notebook source
import os

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
db_name = username.split('.')[0].replace("@","_") + "_nyc"
sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
sql(f"USE {db_name}")
os.environ['username'] = username
os.environ['db_name'] = db_name
print(f"...username: '{username}', db_name: '{db_name}' (create & use)")

# COMMAND ----------

etl_dir = "/datasets/nyc"
etl_dir_fuse = "/dbfs" + etl_dir
os.environ['etl_dir'] = etl_dir
os.environ['etl_dir_fuse'] = etl_dir_fuse
print(f"...etl_dir: '{etl_dir}', etl_dir_fuse: '{etl_dir_fuse}' (create)")
