// Databricks notebook source
val osmFile = "/dbfs/datasets/graphhopper/osm/us-northeast/us-northeast-latest.osm.pbf"

// COMMAND ----------

import com.graphhopper.GHRequest
import com.graphhopper.GHResponse
import com.graphhopper.GraphHopper
import com.graphhopper.ResponsePath
import com.graphhopper.config.CHProfile
import com.graphhopper.config.LMProfile
import com.graphhopper.config.Profile
import com.graphhopper.routing.weighting.custom.CustomProfile
import com.graphhopper.util._
import com.graphhopper.util.shapes.GHPoint

// COMMAND ----------

val hopper = new GraphHopper()
hopper.setOSMFile(osmFile)
hopper.setGraphHopperLocation("target/routing-graph-cache")
hopper.setProfiles(new Profile("car").setVehicle("car").setWeighting("fastest").setTurnCosts(false))
hopper.getCHPreparationHandler().setCHProfiles(new CHProfile("car"))
hopper.importOrLoad()

// COMMAND ----------

dbutils.fs.ls("dbfs:/datasets/graphhopper/osm/us-northeast/")

// COMMAND ----------

dbutils.fs.ls("file:/databricks/driver/")

// COMMAND ----------

dbutils.fs.ls("file:/databricks/driver/target/routing-graph-cache")

// COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/target/routing-graph-cache", "dbfs:/datasets/graphhopper/osm/us-northeast/graphHopperData/", recurse=true)

// COMMAND ----------

dbutils.fs.ls("dbfs:/datasets/graphhopper/osm/us-northeast/")

// COMMAND ----------


