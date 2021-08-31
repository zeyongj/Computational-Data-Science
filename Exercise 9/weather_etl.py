import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('weather ETL').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.4' # make sure we have Spark 2.4+

observation_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.StringType()),
    types.StructField('observation', types.StringType()),
    types.StructField('value', types.IntegerType()),
    types.StructField('mflag', types.StringType()),
    types.StructField('qflag', types.StringType()),
    types.StructField('sflag', types.StringType()),
    types.StructField('obstime', types.StringType()),
])


def main(in_directory, out_directory):

    weather = spark.read.csv(in_directory, schema=observation_schema)

    # TODO: finish here.
    # The following codes are adapted from https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.Column.isNull.html .
    from pyspark.sql import Row
    df = weather.filter(weather['qflag'].isNull())

    # The following code is adapted from https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.Column.startswith.html .
    df = df.filter(df['station'].startswith('CA'))

    df = df.filter(df['observation'] == 'TMAX')

    # The following code is adapted from https://sparkbyexamples.com/pyspark/pyspark-withcolumn/ .
    df = df.withColumn("tmax",(df["value"]/ 10) )

    # The following code is adapted from https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.select.html .
    cleaned_data = df.select('station', 'date', 'tmax')  

    cleaned_data.write.json(out_directory, compression='gzip', mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
