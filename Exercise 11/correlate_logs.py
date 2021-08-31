# README
# To run the program using a small dataset, type "spark-submit correlate_logs.py nasa-logs-1" on the terminal, and the result is "r = 0.630006 r^2 = 0.396907".
# To try a larger dataset, type "spark-submit correlate_logs.py nasa-logs-2" on the terminal, and the result is "r = 0.928466 r^2 = 0.862048".

import sys
from pyspark.sql import SparkSession, functions, types, Row
import re

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        # TODO, the following codes are inspired from the hints: https://coursys.sfu.ca/2021su-cmpt-353-d1/pages/Exercise11 .
        hn = m.group(1)
        nb = int(m.group(2))
        return Row(hostname = hn, bytes = nb) # Number of bytes MUST be integer.
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it. # Yes, this is NONE.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects, the following codes are inspired from the hints: https://coursys.sfu.ca/2021su-cmpt-353-d1/pages/Exercise11 .
    # 1. Get the data out of the files into a DataFrame where you have the hostname and number of bytes for each request. Do this using an RDD operation. 
    row = log_lines.map(line_to_row)
    row = row.filter(not_none)
    return row 


def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    # TODO: calculate r.
    import math
    from pyspark.sql.functions import col
    # 2. Group by hostname; get the number of requests and sum of bytes transferred, to form a data point.
    hostnames = logs.groupBy('hostname')
    count_requests = hostnames.count()
    sum_request_bytes = hostnames.sum('bytes')
    data = sum_request_bytes.join(count_requests, on = 'hostname', how = 'left')
    # print(data.head(5))
    # Output: Record 1: Row(hostname='grimnet23.idirect.com', sum(bytes)=3884, count=5)
    # Attributes: hostname, sum(bytes), count

    # data = data.withColumnRenamed('count', 'xi')
    # data = data.withColumnRenamed('sum(bytes)', 'yi')

    # 3. Produce six values. Add these to get the six sums.
    # The following codes are adapted from https://sparkbyexamples.com/pyspark/pyspark-withcolumn/ .
    data = data.withColumn('xi', data['count'])
    data = data.withColumn('yi', data['sum(bytes)'])
    data = data.withColumn('xiyi', data['xi'] * data['yi'])
    data = data.withColumn('xi^2', data['xi'] * data['xi'])
    data = data.withColumn('yi^2', data['yi'] * data['yi'])

    data = data.select('xi', 'yi', 'xiyi', 'xi^2', 'yi^2')
    n = data.count()
    # print('n = %d' % n)
    
    # The following codes are inspried from the hints: https://coursys.sfu.ca/2021su-cmpt-353-d1/pages/Exercise11 .
    groups = data.groupBy()
    six_sums = groups.sum()
    # six_sums.show()
    # Columns: |sum(xi)| sum(yi)|sum(xiyi)|sum(xi^2)|sum(yi^2)|.
    
    sum_xi =  six_sums.first()[0]
    sum_yi =  six_sums.first()[1]
    sum_xiyi =  six_sums.first()[2]
    sum_xi_squared =  six_sums.first()[3]
    sum_yi_squared =  six_sums.first()[4]
    
    # 4. Calculate the final value of r.
    # As required in the instruction:
    numerator = (n * sum_xiyi) - (sum_xi * sum_yi)
    denumerator1 = (n * sum_xi_squared) - (sum_xi * sum_xi)
    denumerator2 = (n * sum_yi_squared) - (sum_yi * sum_yi)
    denumerator1_sqrt = math.sqrt(denumerator1)
    denumerator2_sqrt = math.sqrt(denumerator2)
    denumerator = denumerator1_sqrt * denumerator2_sqrt
    r = numerator / denumerator # TODO: it isn't zero.
    print("r = %g\nr^2 = %g" % (r, r**2)) 


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
