import json
import numpy as np
import pdb
import naiveMatch
#import bm25 
import sys
import string
import unicodedata
from math import log
import re
rx = re.compile(r"[\W]")


from pyspark import SparkContext, SQLContext
from pyspark.conf import SparkConf
import pyspark
sc = SparkContext(appName="nel-system")
sess = pyspark.sql.SparkSession.builder.appName("nel-system").getOrCreate()
sqlContext = SQLContext(sc)



def loadIndexes(x):
    x_dic = json.loads(x)
    indexes = [int(xx) for xx in x_dic["features"].keys()]
    return indexes

kbFileName = sys.argv[1]
kb_rdd = sc.textFile(kbFileName).flatMap(lambda x: loadIndexes(x)).map(lambda x: (x,1))

print(kb_rdd.collect()[3:5])

N = float(kb_rdd.count()) 
X = kb_rdd.aggregateByKey(0,lambda x,y:x+y,lambda x,y:x+y).map(lambda x : (x[0], np.log(N/x[1])))

X.saveAsTextFile("idf_diags.json")
#X = x.collect()
#
#dic_idf_diags = {}
#for xx in X:
#    dic_idf_diags[xx[0]] = np.log(N/xx[1])
#
#
#json.dump(dic_idf_diags, open("idf_diags.json", "w"))
    
