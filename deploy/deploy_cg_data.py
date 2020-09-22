import time
import sys
sys.path.append('/recommender-data')

from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer

from sys import stdout
from pipeline.file_utils import *
from pipeline.dataframe_utils import *
from google.cloud import storage

# initialize spark
spark_session = SparkSession.builder.\
    appName("sample").\
    config("spark.jars", "/usr/lib/spark/jars/gcs-connector-latest-hadoop2.jar").\
    config('spark.executor.memory', '6g').\
    config('spark.executor.cores', '2').\
    config('spark.driver.memory', '2g').\
    getOrCreate()
spark_session._jsc.hadoopConfiguration().set('fs.gs.impl', 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem')
spark_session._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.json.keyfile",
                                             "/recommender-data/.gcp/credentials/VM Intro-30fcfec18d87.json")
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
max_user_id, max_product_id = np.max(list(user_id_mapping.values())), np.max(list(product_id_mapping.values()))
max_values = ratings.select(max('user_id').alias('user_id'), max('product_id').alias('product_id')).collect()[0].asDict()
save_pickle(max_values, os.path.join(model_path, 'embedding_max_values.pkl'))
save_blob = storage_bucket.blob(os.path.join(model_path, 'embedding_max_values.pkl'))
save_blob.upload_from_filename(os.path.join(model_path, 'embedding_max_values.pkl'))

# create window spec for user touch windows
spark_session.udf.register('get_last_n_elements', get_last_n_elements)
stdout.write('DEBUG: Creating touched windows ...\n')
ratings = ratings.withColumn('timestamp', col('timestamp').cast('long'))
user_window_preceding = Window.partitionBy('user_id').orderBy(asc('timestamp')).rowsBetween(-6, -1)
user_window_present_reversed = Window.partitionBy('user_id').orderBy(desc('timestamp'))
ratings = ratings.repartition(col('user_id'))

stdout.write('DEBUG: Building all touched dictionary ...')
all_touched = ratings.groupby('user_id').agg(collect_list('product_id').alias('all_touched_product_id'))
all_touched_dict = all_touched.rdd.map(lambda row: row.asDict()).collect()
all_touched_dict = dict([(elem['user_id'], elem['all_touched_product_id']) for elem in all_touched_dict])
broadcasted_touched_dict = spark_session.sparkContext.broadcast(all_touched_dict)

average_touched_items = np.mean([len(elems) for user, elems in all_touched_dict.items()])
stdout.write('EVALUATION: Average touched items (non-unique) by user is ' + str(average_touched_items))

ratings = ratings.withColumn('touched_product_id', collect_list(col('product_id')).over(user_window_preceding))
ratings = ratings.withColumn('touched_product_id', sort_array('touched_product_id'))

stdout.write('Reconfigure remaining ratings for negative sampling ...')
num_products = int(max_product_id + 1)
ratings = ratings.drop('timestamp')
ratings.persist()

# negative sample
stdout.write('DEBUG: Beginning negative sampling ... \n')
negative_sampling_distributed = negative_sampling_distributed_f(broadcasted_touched_dict)
spark_session.udf.register('negative_sampling_distributed', negative_sampling_distributed)
negative_sampling_distributed_udf = udf(negative_sampling_distributed, ArrayType(StringType()))
ratings_negative = ratings.withColumn(
    'negatives', negative_sampling_distributed_udf('user_id', lit(num_products), lit(3))
)

ratings_negative = ratings_negative.\
    drop('product_id').\
    withColumn('product_id', explode('negatives')).\
    drop('negatives')
ratings_negative = ratings_negative.\
    drop('target').\
    withColumn('target', lit(0))
ratings_negative.persist()

ratings = ratings.drop('negatives')
ratings_all = ratings.unionByName(ratings_negative)
ratings_all.show()

stdout.write('DEBUG: Beginning stratified split ...')
ratings_all = ratings_all.select('user_id', 'product_id', 'touched_product_id', 'target')
train_df, val_df = stratified_split_distributed(ratings_all, 'target', spark_session)
duration = time.time() - start_time
stdout.write('BENCHMARKING: Runtime was ' + str(duration) + '\n')

stdout.write('DEBUG: Converting dataframes to pandas ...' + '\n')
train_pd = train_df.toPandas()
train_pd.to_csv(os.path.join(model_path, 'train.csv'), index=False)
save_blob = storage_bucket.blob(os.path.join(model_path, 'train.csv'))
save_blob.upload_from_filename(os.path.join(model_path, 'train.csv'), timeout=7200)

val_pd = val_df.toPandas()
val_pd.to_csv(os.path.join(model_path, 'validation.csv'), index=False)
save_blob = storage_bucket.blob(os.path.join(model_path, 'validation.csv'))
save_blob.upload_from_filename(os.path.join(model_path, 'validation.csv'), timeout=7200)

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
