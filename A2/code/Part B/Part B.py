# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, avg

# Initialize Spark session
spark = SparkSession.builder.appName("RecommenderSystem").getOrCreate()


# COMMAND ----------

# Load the data
data = spark.read.csv("/FileStore/tables/movies.csv", header=True, inferSchema=True)

# Display the data
display(data)


# COMMAND ----------

# Describe the data
display(data.describe())


# COMMAND ----------

# Calculate the average rating for each movie
avg_movie_ratings = data.groupBy("movieId").agg(avg("rating").alias("avg_rating"))

# Get the top 12 movies with the highest average ratings
top_12_movies = avg_movie_ratings.orderBy(desc("avg_rating")).limit(12)

# Show the top 12 movies
display(top_12_movies)


# COMMAND ----------

# Calculate the average rating provided by each user
avg_user_ratings = data.groupBy("userId").agg(avg("rating").alias("avg_rating"))

# Get the top 12 users who provided the highest average ratings
top_12_users = avg_user_ratings.orderBy(desc("avg_rating")).limit(12)

# Show the top 12 users
display(top_12_users)


# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# COMMAND ----------

# Function to train ALS model and evaluate performance
def train_and_evaluate(data, train_ratio, test_ratio):
    # Split the data into training and test sets
    (training, test) = data.randomSplit([train_ratio, test_ratio])

    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

    # Train the model
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error for {train_ratio*100}/{test_ratio*100} split = {rmse}")

    return rmse


# COMMAND ----------

# Perform training and evaluation for different splits
rmse_70_30 = train_and_evaluate(data, 0.7, 0.3)

# COMMAND ----------

rmse_80_20 = train_and_evaluate(data, 0.8, 0.2)

# COMMAND ----------

# Function to train ALS model and evaluate performance using RMSE and MAE
def train_and_evaluate(data, train_ratio, test_ratio):
    # Split the data into training and test sets
    (training, test) = data.randomSplit([train_ratio, test_ratio])

    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

    # Train the model
    model = als.fit(training)

    # Evaluate the model by computing the RMSE and MAE on the test data
    predictions = model.transform(test)
    
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator_rmse.evaluate(predictions)
    
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
    mae = evaluator_mae.evaluate(predictions)
    
    print(f"Evaluation metrics for {train_ratio*100}/{test_ratio*100} split:")
    print(f"Root-mean-square error (RMSE) = {rmse}")
    print(f"Mean absolute error (MAE) = {mae}")

    return rmse, mae


# COMMAND ----------

# Perform training and evaluation for different splits
rmse_70_30, mae_70_30 = train_and_evaluate(data, 0.7, 0.3)
rmse_80_20, mae_80_20 = train_and_evaluate(data, 0.8, 0.2)

# COMMAND ----------

# Compare results
print("\nComparison of RMSE and MAE for different splits:")
print(f"70/30 split - RMSE: {rmse_70_30}, MAE: {mae_70_30}")
print(f"80/20 split - RMSE: {rmse_80_20}, MAE: {mae_80_20}")

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# COMMAND ----------

# Split the data into training and test sets (80/20 split)
(training, test) = data.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

# Define the evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")


# COMMAND ----------

# Define the parameter grid for tuning
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 50, 100]) \
    .addGrid(als.maxIter, [10, 15, 20]) \
    .addGrid(als.regParam, [0.01, 0.1, 1.0]) \
    .build()

# Create a CrossValidator
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# COMMAND ----------

# Run cross-validation, and choose the best set of parameters
cvModel = crossval.fit(training)

# COMMAND ----------

# Make predictions on the test data
predictions = cvModel.transform(test)

# COMMAND ----------

# Evaluate the model
rmse = evaluator.evaluate(predictions)
print(f"Best Model Root-mean-square error (RMSE) = {rmse}")

# COMMAND ----------

# Show the best parameters
best_model = cvModel.bestModel
print(f"Best rank: {best_model._java_obj.parent().getRank()}")
print(f"Best maxIter: {best_model._java_obj.parent().getMaxIter()}")
print(f"Best regParam: {best_model._java_obj.parent().getRegParam()}")

# COMMAND ----------

# Get the top 12 movie recommendations for user ID 10 and user ID 12
user_10_recs = best_model.recommendForUserSubset(training.filter(training.userId == 10), 12)
user_12_recs = best_model.recommendForUserSubset(training.filter(training.userId == 12), 12)


# COMMAND ----------

# Show recommendations for user ID 10
print("Top 12 movie recommendations for user ID 10:")
user_10_recs.show(truncate=False)

# COMMAND ----------

# Show recommendations for user ID 12
print("Top 12 movie recommendations for user ID 12:")
user_12_recs.show(truncate=False)
