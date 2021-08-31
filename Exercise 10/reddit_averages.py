import sys
# import pandas
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+


comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])


def main(in_directory, out_directory):
    # comments = spark.read.json(in_directory) # Without schema.
    comments = spark.read.json(in_directory, schema=comments_schema) # With schema.

    # TODO: calculate averages, sort by subreddit. Sort by average score and output that too.
    subreddit = comments.groupBy('subreddit')
    from pyspark.sql.functions import col, avg
    # The following code is adapted from https://stackoverflow.com/questions/32550478/pyspark-take-average-of-a-column-after-using-filter-function .
    averages = subreddit.agg(avg(comments['score']))

    # Without cache.
    averages = averages.cache() # With cashe, adapted from https://coursys.sfu.ca/2021su-cmpt-353-d1/pages/Exercise10 , as required.

    # The following codes are adapted from https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.sort.html .
    averages_by_subreddit = averages.sort('subreddit', ascending = True)
    averages_by_score = averages.sort('avg(score)', ascending = False)
    

    # Uncomment the following codes, as required in the instruction.
    averages_by_subreddit.write.csv(out_directory + '-subreddit', mode='overwrite')
    averages_by_score.write.csv(out_directory + '-score', mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
