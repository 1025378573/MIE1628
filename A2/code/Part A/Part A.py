# Databricks notebook source
# Import necessary libraries
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("OddEvenCount").getOrCreate()

# COMMAND ----------

# Read the integer.txt file
# Adjust the path according to your file location in Databricks
df = spark.read.text("/FileStore/tables/integer.txt")

# COMMAND ----------

# Show the content of the dataframe
df.show()

# COMMAND ----------

# Convert the dataframe to RDD and extract the integers
numbers_rdd = df.rdd.map(lambda row: int(row[0]))

# COMMAND ----------

numbers_rdd.collect()

# COMMAND ----------


# Function to determine if a number is odd or even
def odd_even(num):
    if num % 2 == 0:
        return ("Even", 1)
    else:
        return ("Odd", 1)

# COMMAND ----------

# Map the numbers to odd/even and reduce by key to count them
count_rdd = numbers_rdd.map(odd_even).reduceByKey(lambda a, b: a + b)

# COMMAND ----------

# Collect the result
result = count_rdd.collect()


# COMMAND ----------

# Print the result
for key, count in result:
    print(f"{key}: {count}")

# COMMAND ----------

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, sum as spark_sum

# COMMAND ----------

# Create Spark session
spark = SparkSession.builder.appName("SalarySumPerDepartment").getOrCreate()

# COMMAND ----------

# Read the salary.txt file
# Adjust the path according to your file location in Databricks
df = spark.read.text("/FileStore/tables/salary.txt")

# COMMAND ----------

# Split the lines into department and salary
split_col = split(df['value'], ' ')
df = df.withColumn('department', split_col.getItem(0))
df = df.withColumn('salary', split_col.getItem(1).cast('float'))

# COMMAND ----------

display(df)

# COMMAND ----------

# Group by department and calculate the sum of salaries
result = df.groupBy('department').agg(spark_sum('salary').alias('salary_sum'))

# COMMAND ----------

result.show()

# COMMAND ----------

display(result)

# COMMAND ----------

# Create Spark session
spark = SparkSession.builder.appName("WordCount").getOrCreate()

# Read the shakespeare.txt file
# Adjust the path according to your file location in Databricks
df = spark.read.text("/FileStore/tables/shakespeare_1.txt")

# COMMAND ----------

display(df)

# COMMAND ----------

# Convert the dataframe to RDD
lines_rdd = df.rdd.map(lambda row: row[0])

# COMMAND ----------

lines_rdd.collect()

# COMMAND ----------

# List of words to count
words_to_count = ["Shakespeare", "When", "Lord", "Library", "GUTENBERG", "WILLIAM", "COLLEGE", "WORLD"]

# Function to count specified words in a line
def word_count(line, words):
    line_words = line.split()
    word_counts = []
    for word in words:
        count = line_words.count(word)
        if count > 0:
            word_counts.append((word, count))
    return word_counts

# COMMAND ----------

# FlatMap the lines RDD to count occurrences of the specified words
word_counts_rdd = lines_rdd.flatMap(lambda line: word_count(line, words_to_count))

# COMMAND ----------

# Reduce by key to sum the counts for each word
word_counts = word_counts_rdd.reduceByKey(lambda a, b: a + b)

# COMMAND ----------


# Collect the result
result = word_counts.collect()

# COMMAND ----------


# Print the result
for word, count in result:
    print(f"{word}: {count}")

# COMMAND ----------

from pyspark.sql.functions import explode, split, col
from pyspark.sql.types import StringType

# COMMAND ----------

# Create Spark session
spark = SparkSession.builder.appName("TopBottomWordCount").getOrCreate()

# COMMAND ----------

# Split lines into words
words_df = df.select(explode(split(col("value"), "\\s+")).alias("word"))

# COMMAND ----------

# Remove any empty strings resulting from multiple spaces
words_df = words_df.filter(words_df.word != "")

# COMMAND ----------

display(words_df)

# COMMAND ----------

# Count the occurrences of each word
word_counts_df = words_df.groupBy("word").count()

# COMMAND ----------

# Sort the words by count in descending order for the top 15
top_15_words = word_counts_df.orderBy(col("count").desc()).limit(15)

# Sort the words by count in ascending order for the bottom 15
bottom_15_words = word_counts_df.orderBy(col("count").asc()).limit(15)

# COMMAND ----------

# Show the top 15 words
print("Top 15 words:")
top_15_words.show()


# COMMAND ----------

# Show the bottom 15 words
print("Bottom 15 words:")
bottom_15_words.show()
