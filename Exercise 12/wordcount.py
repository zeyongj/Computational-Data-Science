# README
# To run the program, type "spark-submit wordcount.py wordcount-1 output" in the terminal.

from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql.functions import split, explode, col

import string, re, sys

# I use a method different from the previous hints.
# Judge whether the inputing parameters are valid.
# It must specify two inputing parameters.
# The first one is the text file to being handlered.
# And the last one is the path used to save the outputing result.

if 3 != len(sys.argv):
    print("Usage: invalid input parameters")
    exit(0)

# Obtain the input file.
input_file = sys.argv[1]
print("input file is: {}".format(input_file))
# Otain the output file path.
output_path = sys.argv[2]
print("output path is: {}".format(output_path))

# Initialize the spark session.
spark = SparkSession.builder.appName("wordcount").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

# Regex expression used to filter all the unncessary word.
wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)

# Read text file.
lines = spark.read.text(input_file)

# Split the text row with the specify regex expression.
words = lines.select(explode(split(lines.value, wordbreak)).alias("word"))

# Filter all the no-empty column.
words = words.filter(words['word'] != '')

# Create temporary view table 'temp_word_count'.
words.createTempView("temp_word_count")

# Count the word and sort them by its value.
df = spark.sql("select word, count(word) as count from temp_word_count group by word order by count(word) desc")

# words = words.groupBy('word').agg(functions.count(words['word']).alias('count'))
# df = words.sort(words['count'], ascending = False)

# Output the statistic result to a csv file.
# df.coalesce(1).write.option("header", "true").csv(output_path) # I think coalesce is safe, but I am not sure. 
# df.write.csv(output_path, mode = 'overwrite')
# df.show()

# The following code is adapted from https://stackoverflow.com/questions/35861099/overwriting-a-spark-output-using-pyspark .
# df.coalesce(1).write.mode('overwrite').option("header", "true").csv(output_path)
df.write.mode('overwrite').option("header", "true").csv(output_path)

