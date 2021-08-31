import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
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
]) # I did not change schema.


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema)

    # TODO
    # 1. Calculate the average score for each subreddit, as before.
    # The following codes are taken directly from my last exercise.
    subreddit = comments.groupBy('subreddit')
    from pyspark.sql.functions import col, avg
    # The following code is adapted from https://stackoverflow.com/questions/32550478/pyspark-take-average-of-a-column-after-using-filter-function .
    averages = subreddit.agg(avg(comments['score']).alias('average'))

    averages = averages.cache() # According to my finding in the last exercise, caching after the group by function and before the join function is the best choice.

    # 2. Exclude any subreddits with average score <= 0.
    averages = averages.drop(averages['average'] <= 0)

    # 3. Join the average score to the collection of all comments. Divide to get the relative score.
    averages = averages.join(comments, on = 'subreddit', how = 'left')

    # The following code is adapted from https://sparkbyexamples.com/pyspark/pyspark-withcolumn/ .
    averages = averages.withColumn("relative_score", (averages['score'] / averages['average']) )

    averages1 = averages.groupBy('subreddit').agg(functions.max(averages['relative_score']).alias("relative_score"))

    # 4. Determine the max relative score for each subreddit.
    max_relative_score = comments.groupBy('subreddit').agg(functions.max(comments['score']).alias("score"))
    max_relative_score = max_relative_score.cache()

    # 5. Join again to get the best comment on each subreddit: we need this step to get the author.
    max_relative_score = max_relative_score.join(comments, on = ['score', 'subreddit'], how = 'left')

    best_author = max_relative_score.join(averages1, on = 'subreddit', how = 'left')
    best_author = best_author.select('subreddit', 'author', 'relative_score')
    best_author = best_author.withColumn('rel_score', best_author['relative_score']) 
    best_author = best_author.select('subreddit', 'author', 'rel_score')
    best_author = best_author.sort('subreddit', 'author', ascending = True, key = lambda x: x.str.lower()) # Adapted from https://stackoverflow.com/questions/29898090/pandas-sort-with-capital-letters .
    # best_author.show()

    best_author.write.json(out_directory, mode = 'overwrite') # Uncomment this line of code, as required in the instruction.


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
