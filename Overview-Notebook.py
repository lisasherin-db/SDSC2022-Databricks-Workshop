# Databricks notebook source
# MAGIC %md # SDSC2022 NYC - Databricks Workshop
# MAGIC 
# MAGIC > We presented this "Building a Flexible Geospatial Lakehouse on the Databricks Platform" material to workshop attendees at CARTO's 2022 [Spatial Data Science Conference](https://spatial-data-science-conference.com/2022/newyork/#workshops) in NYC. 
# MAGIC 
# MAGIC In addition to using Databricks platform, this material also uses Databricks Labs Mosaic project [[docs](https://databrickslabs.github.io/mosaic/index.html) | [repo](https://github.com/databrickslabs/mosaic) | [announcement blog](https://www.databricks.com/blog/2022/05/02/high-scale-geospatial-processing-with-mosaic.html)]. 

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.image as img
plt.rcParams["figure.figsize"] = [16, 9] # <-- inches
plt.rcParams["figure.autolayout"] = True

# COMMAND ----------

# MAGIC %md ## Presenters

# COMMAND ----------

plt.imshow(img.imread("Resources/presenters.png"), aspect='auto')
plt.show()

# COMMAND ----------

# MAGIC %md ## Why H3?
# MAGIC 
# MAGIC > More [[docs](https://docs.databricks.com/spark/latest/spark-sql/language-manual/sql-ref-functions-builtin.html#h3-geospatial-functions) | [announcement blog](https://www.databricks.com/blog/2022/09/14/announcing-built-h3-expressions-geospatial-processing-and-analytics.html)]

# COMMAND ----------

plt.imshow(img.imread("Resources/why_h3.png"), aspect='auto')
plt.show()

# COMMAND ----------

# MAGIC %md ## Example 01
# MAGIC 
# MAGIC > See Directory __NYC H3 Analysis__

# COMMAND ----------

plt.imshow(img.imread("Resources/1a_proximity.png"), aspect='auto')
plt.show()

# COMMAND ----------

plt.imshow(img.imread("Resources/1b_proximity.png"), aspect='auto')
plt.show()

# COMMAND ----------

# MAGIC %md ## Example 02
# MAGIC 
# MAGIC > See Directory __Isochrones__

# COMMAND ----------

plt.imshow(img.imread("Resources/2a_site_selection.png"), aspect='auto')
plt.show()

# COMMAND ----------

plt.imshow(img.imread("Resources/2b_site_selection.png"), aspect='auto')
plt.show()

# COMMAND ----------

# MAGIC %md ## Example 03
# MAGIC 
# MAGIC > See Directory __KNN__

# COMMAND ----------

plt.imshow(img.imread("Resources/3a_spatial_neighbors.png"), aspect='auto')
plt.show()

# COMMAND ----------

plt.imshow(img.imread("Resources/3b_spatial_neighbors.png"), aspect='auto')
plt.show()
