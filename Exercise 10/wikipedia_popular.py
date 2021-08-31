# The following codes are inspried and adapted from reddit_averages.py .
# To run the program, please enter "spark-submit --master=local[1] wikipedia_popular.py pagecounts-1 output" in the terminal.
# The output would be stored in the folder of "output most_frequently_accessed_page".
import sys
# import pandas
from pyspark.sql import SparkSession, functions, types
import re

spark = SparkSession.builder.appName('wikipedia popular').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+


wiki_schema = types.StructType([
    types.StructField('language', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('requested_times', types.LongType()),
    types.StructField('size_bytes', types.LongType()),
])

# The following function is adapted from https://hang-hu.github.io/pyspark/2019/01/17/udf.html .
def myHour(path):
    return re.search('\d{8}-\d{2}', path).group(0)


def main(in_directory, out_directory):
    # No cache: 12.539s, 13.019s, 13.512s, 13.664s, 13.549s, AVG = 13.2796s
    # wiki = spark.read.csv(in_directory, sep = " ").withColumn('filename', functions.input_file_name()) # Without schema.
    wiki = spark.read.csv(in_directory, schema = wiki_schema, sep = " ").withColumn('filename', functions.input_file_name()) # With schema, adapted from https://coursys.sfu.ca/2021su-cmpt-353-d1/pages/Exercise10 , as required.
    english_wikipedia_pages = (wiki['language'] == 'en')
    not_most_frequent_page = (wiki['title'] != 'Main_Page')
    
    wiki = wiki.filter(english_wikipedia_pages)
    wiki = wiki.filter(not_most_frequent_page)
    # wiki = wiki.filter(wiki['language'] == 'en')
    # wiki = wiki.filter(wiki['title'] != 'Main_Page')
    # The following code is adapted from https://sparkbyexamples.com/spark/spark-filter-startswith-endswith-examples/ .
    wiki = wiki.drop(wiki.title.startswith('Special:'))
    
    # First place of cache
    # wiki = wiki.cache()
    # Time: 12.950s, 13.305s, 13.092s, 13.290s, 13.615s, AVG = 13.2504s
    
    path_to_hour = functions.udf(myHour, returnType = types.StringType())
    wiki = wiki.withColumn('hour', path_to_hour(wiki['filename']))
    
    # Second place of cache
    # wiki = wiki.cache()
    # Time: 12.616s, 12.552s, 12.540s, 12.914s, 13.137s, AVG = 12.7518s

    hourly_wiki = wiki.groupby('hour')
    # The following code is adapted from https://stackoverflow.com/questions/33516490/column-alias-after-groupby-in-pyspark .
    from pyspark.sql.functions import max
    max_wiki = hourly_wiki.agg(max(wiki['requested_times']).alias('requested_times'))
    
    # ***BEST: Third place of cache***
    wiki = wiki.cache() # Time: 12.378s, 12.687s, 12.790s, 12.693s, 12.559s. AVG = 12.6214s (Good if only using cache of max_wiki) 
    # LOWEST RUNNING TIME = 12.378s
    # LOWEST AVERAGE RUNNING TIME = 12.6214s
    # max_wiki = max_wiki.cache() # Time: 12.890s (Not good if only using cache of max_wiki)
    # Time: 12.771s (Not good if using both)
    
    # The following codes are adapted from https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#joining-on-index .
    # import pandas as pd
    max_wiki = max_wiki.join(wiki, on=["requested_times", "hour"], how = 'left')

    # MEANINGLESS: Fourth place of cache
    # wiki = wiki.cache() # Time: 12.495s, 12.463s, 12.849s, 12.732s, 12.754s. AVG = 12.6586s (Good if only using cache of max_wiki)
    # max_wiki = max_wiki.cache() # Time: 13.261s (Not good if only using cache of max_wiki)
    # Time: 12.897s (Not good if using both)
    
    max_wiki = max_wiki.select('hour', 'title', 'requested_times')
    max_wiki = max_wiki.sort('hour', 'title', ascending = True)
   
    max_wiki.write.csv(out_directory + ' most_frequently_accessed_page', mode = 'overwrite')

    # max_wiki.show()



if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
