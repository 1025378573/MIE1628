# Databricks notebook source
# Step 1: Import the necessary libraries
import urllib.request

# Step 2: Download the data and store it in a temporary location
urllib.request.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "/tmp/kddcup_data.gz")

# Step 3: Move the data from the temporary location to the Databricks filesystem
dbutils.fs.mv("file:/tmp/kddcup_data.gz", "dbfs:/kdd/kddcup_data.gz")

# Step 4: Display the contents of the directory to confirm the data is moved
display(dbutils.fs.ls("dbfs:/kdd"))


# COMMAND ----------

# Step 1: Load the data from the Databricks filesystem into an RDD
rdd = sc.textFile("dbfs:/kdd/kddcup_data.gz")

# Step 2: Print 10 values from the RDD
print("First 10 values of the RDD:")
for line in rdd.take(10):
    print(line)

# Step 3: Verify the type of data structure
print("\nType of the data structure:", type(rdd))


# COMMAND ----------


# Step 1: Split each line into a list of features
rdd_split = rdd.map(lambda line: line.split(","))

# Step 2: Show the total number of features (columns)
num_features = len(rdd_split.first())
print("Total number of features (columns):", num_features)

# Step 3: Print the first 10 rows of the split data
print("First 10 rows of the split data:")
for row in rdd_split.take(10):
    print(row)



# COMMAND ----------

# Define the indices of the required columns based on the KDD Cup 99 dataset description
required_columns = {
    "duration": 0,
    "protocol_type": 1,
    "service": 2,
    "flag": 3,
    "src_bytes": 4,
    "dst_bytes": 5,
    "label": 41  # 'label' is the last column
}

# Extract the specified columns
rdd_extracted = rdd_split.map(lambda line: (
    line[required_columns["duration"]],
    line[required_columns["protocol_type"]],
    line[required_columns["service"]],
    line[required_columns["flag"]],
    line[required_columns["src_bytes"]],
    line[required_columns["dst_bytes"]],
    line[required_columns["label"]]
))

# Convert the RDD to a DataFrame
from pyspark.sql import Row

# Create a DataFrame from the RDD
df_extracted = rdd_extracted.map(lambda row: Row(
    duration=row[0],
    protocol_type=row[1],
    service=row[2],
    flag=row[3],
    src_bytes=row[4],
    dst_bytes=row[5],
    label=row[6]
)).toDF()

# Print the schema of the DataFrame
df_extracted.printSchema()

# Display the first 10 rows of the DataFrame
display(df_extracted.limit(10))


# COMMAND ----------

# Step 1: Aggregate the data based on protocol_type and service

# Aggregate based on protocol_type
protocol_type_counts = df_extracted.groupBy("protocol_type").count().orderBy("count", ascending=True)

# Aggregate based on service
service_counts = df_extracted.groupBy("service").count().orderBy("count", ascending=True)

# Step 2: Show the results in ascending order

# Show the results for protocol_type
display(protocol_type_counts)

# Show the results for service
display(service_counts)

# Step 3: Plot the bar graphs

import matplotlib.pyplot as plt

# Convert to Pandas DataFrame for plotting
protocol_type_counts_pd = protocol_type_counts.toPandas()
service_counts_pd = service_counts.toPandas()

# Plot the bar graph for protocol_type
plt.figure(figsize=(10, 5))
plt.bar(protocol_type_counts_pd['protocol_type'], protocol_type_counts_pd['count'], color='blue')
plt.xlabel('Protocol Type')
plt.ylabel('Number of Connections')
plt.title('Total Number of Connections Based on Protocol Type')
plt.xticks(rotation=45)
plt.show()

# Plot the bar graph for service
plt.figure(figsize=(15, 5))
plt.bar(service_counts_pd['service'], service_counts_pd['count'], color='green')
plt.xlabel('Service')
plt.ylabel('Number of Connections')
plt.title('Total Number of Connections Based on Service')
plt.xticks(rotation=90)
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting environment
sns.set(style="whitegrid")

# 1. Distribution of Attack Types (label column)
label_counts = df_extracted.groupBy("label").count().orderBy("count", ascending=False)
label_counts_pd = label_counts.toPandas()

plt.figure(figsize=(10, 5))
sns.barplot(x='label', y='count', data=label_counts_pd, palette='viridis')
plt.xlabel('Label')
plt.ylabel('Number of Connections')
plt.title('Distribution of Attack Types')
plt.xticks(rotation=45)
plt.show()

# 2. Relationship between Source Bytes (src_bytes) and Destination Bytes (dst_bytes) with log scale
# and distinct color for 'normal' points

# Convert to Pandas DataFrame for plotting
data_pd = df_extracted.toPandas()

# Dynamically create a color palette
unique_labels = data_pd['label'].unique()
palette = {label: 'red' if label == 'normal.' else sns.color_palette('viridis', len(unique_labels))[i] for i, label in enumerate(unique_labels)}

plt.figure(figsize=(10, 5))
sns.scatterplot(x='src_bytes', y='dst_bytes', data=data_pd, hue='label', palette=palette)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Source Bytes (log scale)')
plt.ylabel('Destination Bytes (log scale)')
plt.title('Relationship between Source Bytes and Destination Bytes')
plt.legend(loc='upper right')
plt.show()

# 3. Count of Connections based on Flag
flag_counts = df_extracted.groupBy("flag").count().orderBy("count", ascending=False)
flag_counts_pd = flag_counts.toPandas()

plt.figure(figsize=(10, 5))
sns.barplot(x='flag', y='count', data=flag_counts_pd, palette='viridis')
plt.xlabel('Flag')
plt.ylabel('Number of Connections')
plt.title('Count of Connections based on Flag')
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

from pyspark.sql.functions import when

# Create a new label column where 'normal' is 'normal' and everything else is 'attack'
df_extracted = df_extracted.withColumn("new_label", when(df_extracted["label"] == "normal.", "normal").otherwise("attack"))

# Show the updated DataFrame
display(df_extracted.limit(10))


# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

# Convert columns to appropriate types
df_extracted = df_extracted.withColumn("duration", col("duration").cast("double"))
df_extracted = df_extracted.withColumn("src_bytes", col("src_bytes").cast("double"))
df_extracted = df_extracted.withColumn("dst_bytes", col("dst_bytes").cast("double"))

# Index the categorical columns
protocol_indexer = StringIndexer(inputCol="protocol_type", outputCol="protocol_type_indexed")
service_indexer = StringIndexer(inputCol="service", outputCol="service_indexed")
flag_indexer = StringIndexer(inputCol="flag", outputCol="flag_indexed")
label_indexer = StringIndexer(inputCol="new_label", outputCol="label_indexed")

# Assemble the feature columns
feature_assembler = VectorAssembler(inputCols=[
    "duration", 
    "protocol_type_indexed", 
    "service_indexed", 
    "src_bytes", 
    "dst_bytes", 
    "flag_indexed"
], outputCol="features")

# Create a pipeline
pipeline = Pipeline(stages=[protocol_indexer, service_indexer, flag_indexer, label_indexer, feature_assembler])
pipeline_model = pipeline.fit(df_extracted)

# Transform the data
df_transformed = pipeline_model.transform(df_extracted)


# COMMAND ----------

display(df_transformed)

# COMMAND ----------

# Split the data into training and test sets
train_data, test_data = df_transformed.randomSplit([0.8, 0.2], seed=42)


# COMMAND ----------

from pyspark.ml.classification import LinearSVC

# Create a Linear SVM model
svm = LinearSVC(labelCol="label_indexed", featuresCol="features", maxIter=10)

# Train the model
svm_model = svm.fit(train_data)

# Make predictions
predictions = svm_model.transform(test_data)

# Show some predictions
display(predictions.select("new_label", "prediction", "rawPrediction").limit(10))


# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Additional metrics
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)
print(f"F1 Score: {f1_score}")

