import time

from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer

from sys import stdout
from pipeline.file_utils import *
from pipeline.dataframe_utils import *
from google.cloud import storage

# initialize spark
sys.path.append('/Users/michaelma/rush/recommender-data')
spark_session = SparkSession.builder.\
    appName("sample").\
    config("spark.jars", "/usr/local/Cellar/apache-spark/3.0.0/libexec/jars/gcs-connector-latest-hadoop2.jar").\
    config('spark.executor.memory', '2g').\
    config('spark.driver.memory', '2g').\
    getOrCreate()
spark_session._jsc.hadoopConfiguration().set('fs.gs.impl', 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem')
spark_session._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.json.keyfile",
                                             "/Users/michaelma/.gcp/credentials/VM Intro-30fcfec18d87.json")
spark_session._jsc.hadoopConfiguration().set('fs.gs.auth.service.account.enable', 'true')

# setup paths
storage_client = storage.Client()
model_path = 'models/candidate_generation'
model_bucket = 'recommender-amazon-1'
storage_bucket = storage_client.get_bucket(model_bucket)
handle_path(model_path)

# load rating data
rating_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("rating", DoubleType(), True),
    StructField("timestamp", LongType(), True)
])
stdout.write('DEBUG: Reading in data ...\n')
start_time = time.time()
ratings = spark_session.read.csv("gs://recommender-amazon-1/data/ratings/fashion.csv",
                                 header=False,
                                 schema=rating_schema)
ratings = ratings.withColumn("timestamp", to_timestamp(ratings["timestamp"]))

# add relevant data
ratings = ratings.withColumn("target", lit(1))
ratings = ratings.drop("rating")
ratings = ratings.limit(100000)
ratings.persist()

# encode user ids
stdout.write('DEBUG: Encoding categorical values ...\n')
user_id_encoder = StringIndexer().setInputCol('user_id').setOutputCol('encoded_user_id')
user_id_model = user_id_encoder.fit(ratings)
ratings = user_id_model.transform(ratings).withColumn('encoded_user_id',
                                                      col('encoded_user_id').cast('int'))

ratings = ratings.drop('user_id')
ratings = ratings.withColumnRenamed('encoded_user_id', 'user_id')
user_id_mapping = dict([(elem, i) for i, elem in enumerate(user_id_model.labels)])
save_pickle(user_id_mapping, os.path.join(model_path, 'user_id_encoder.pkl'))
save_blob = storage_bucket.blob(os.path.join(model_path, 'user_id_encoder.pkl'))
save_blob.upload_from_filename(os.path.join(model_path, 'user_id_encoder.pkl'))

# encode product ids
product_id_encoder = StringIndexer().setInputCol('product_id').setOutputCol('encoded_product_id')
product_id_model = product_id_encoder.fit(ratings)
ratings = product_id_model.transform(ratings).withColumn('encoded_product_id',
                                                         col('encoded_product_id').cast('int'))
ratings = ratings.drop('product_id')
ratings = ratings.withColumnRenamed('encoded_product_id', 'product_id')
product_id_mapping = dict([(elem, i) for i, elem in enumerate(product_id_model.labels)])
save_pickle(product_id_mapping, os.path.join(model_path, 'product_id_encoder.pkl'))
save_blob = storage_bucket.blob(os.path.join(model_path, 'product_id_encoder.pkl'))
save_blob.upload_from_filename(os.path.join(model_path, 'product_id_encoder.pkl'))

# get max values for embedding dictionaries
stdout.write('Getting max encoded values for embedding dictionaries ...')
max_values = ratings.select(max('user_id').alias('user_id'), max('product_id').alias('product_id')).collect()[0].asDict()
save_pickle(max_values, os.path.join(model_path, 'embedding_max_values.pkl'))
save_blob = storage_bucket.blob(os.path.join(model_path, 'embedding_max_values.pkl'))
save_blob.upload_from_filename(os.path.join(model_path, 'embedding_max_values.pkl'))

# create window spec for user touch windows
spark_session.udf.register('get_last_n_elements', get_last_n_elements)
get_last_n_elements_udf = udf(get_last_n_elements)
stdout.write('DEBUG: Creating touched windows ...\n')
user_window = Window.partitionBy('user_id').orderBy(asc('timestamp'))
ratings = ratings.withColumn('touched_product_id', collect_list(col('product_id')).over(user_window))
ratings = ratings.withColumn('last_6_product_id', get_last_n_elements_udf('touched_product_id', lit(5)))
ratings = ratings.drop('timestamp')

# negative sample
stdout.write('DEBUG: Beginning negative sampling ... \n')
spark_session.udf.register('negative_sampling_distributed', negative_sampling_distributed)
negative_sampling_distributed_udf = udf(negative_sampling_distributed, ArrayType(StringType()))
num_products = ratings.select(countDistinct('product_id')).collect()[0][0]
ratings = ratings.withColumn('touched_product_id', sort_array('touched_product_id'))
ratings = ratings.withColumn(
    'negatives', negative_sampling_distributed_udf('touched_product_id', 'product_id', lit(num_products), lit(3))
)
ratings = ratings.drop('touched_product_id')
ratings.persist()

ratings_negative = ratings.\
    drop('product_id').\
    withColumn('product_id', explode('negatives')).\
    drop('negatives')
ratings_negative = ratings_negative.\
    drop('target').\
    withColumn('target', lit(0))

ratings = ratings.drop('negatives')
ratings_all = ratings.unionByName(ratings_negative)
ratings_all.show()

stdout.write('DEBUG: Beginning stratified split ...')
ratings_all = ratings_all.select('user_id', 'product_id', 'last_6_product_id', 'target')
ratings_all = ratings_all.withColumnRenamed('last_6_product_id', 'touched_product_id')
train_df, val_df = stratified_split_distributed(ratings_all, 'target', spark_session)
duration = time.time() - start_time
stdout.write('BENCHMARKING: Runtime was ' + str(duration) + '\n')

stdout.write('DEBUG: Converting dataframes to numpy matrices ...' + '\n')
train_df.write.parquet(os.path.join('gs://recommender-amazon-1', model_path, 'train'), mode='overwrite')
val_df.write.parquet(os.path.join('gs://recommender-amazon-1', model_path, 'validation'), mode='overwrite')

stdout.write('DEBUG: Saving feature indices ...')
feature_indices = dict([(feature, i) for i, feature in enumerate(ratings_all.schema.fieldNames())])
save_pickle(feature_indices, os.path.join(model_path, 'feature_indices.pkl'))
save_blob = storage_bucket.blob(os.path.join(model_path, 'feature_indices.pkl'))
save_blob.upload_from_filename(os.path.join(model_path, 'feature_indices.pkl'))

stdout.write('DEBUG: Saving feature types ...')
feature_types = dict([(field.name, str(field.dataType)) for field in ratings_all.schema.fields])
save_pickle(feature_types, os.path.join(model_path, 'feature_types.pkl'))
save_blob = storage_bucket.blob(os.path.join(model_path, 'feature_types.pkl'))
save_blob.upload_from_filename(os.path.join(model_path, 'feature_types.pkl'))