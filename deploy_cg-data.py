import time
import sys

from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from sys import stdout
from pipeline.spark_utils import *

def main():
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
    model_path = 'models/luxury-beauty/candidate-generation'
    model_bucket = 'recommender-amazon-1'
    cg_storage = ModelStorage(bucket_name=model_bucket, model_path=model_path)
    handle_path(model_path)

    # load rating data
    rating_schema = StructType([
        StructField("product_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("rating", DoubleType(), True),
        StructField("timestamp", LongType(), True)
    ])
    stdout.write('DEBUG: Reading in data ...\n')
    start_time = time.time()
    ratings = spark_session.read.csv("gs://recommender-amazon-1/data/ratings/luxury-beauty.csv",
                                     header=False,
                                     schema=rating_schema)
    ratings = ratings.withColumn("timestamp", to_timestamp(ratings["timestamp"]))
    ratings = ratings.withColumn("target", lit(1))


    # encode product ids
    ratings, product_id_mapping = encode_column(ratings, 'product_id')
    save_pickle(product_id_mapping, os.path.join(model_path, 'product_id_encoder.pkl'))
    cg_storage.save_file_gcs('product_id_encoder.pkl')

    # encode user ids
    ratings, user_id_mapping = encode_column(ratings, 'user_id')
    save_pickle(user_id_mapping, os.path.join(model_path, 'user_id_encoder.pkl'))
    cg_storage.save_file_gcs('user_id_encoder.pkl')

    # get max values for embedding dictionaries
    stdout.write('Getting max encoded values for embedding dictionaries ...')
    max_user_id, max_product_id = np.max(list(user_id_mapping.values())), np.max(list(product_id_mapping.values()))
    max_values = ratings.select(max('user_id').alias('user_id'), max('product_id').alias('product_id')).collect()[0].asDict()
    save_pickle(max_values, os.path.join(model_path, 'embedding_max_values.pkl'))
    cg_storage.save_file_gcs('embedding_max_values.pkl')

    # create window spec for user touch windows
    spark_session.udf.register('get_last_n_elements', get_last_n_elements)
    stdout.write('DEBUG: Creating touched windows ...\n')
    ratings = ratings.withColumn('timestamp', col('timestamp').cast('long'))
    window_thres = 10
    user_window_preceding = Window.partitionBy('user_id').orderBy(asc('timestamp')).rowsBetween(-window_thres, -1)
    user_window_present = Window.partitionBy('user_id').orderBy(asc('timestamp'))
    ratings = ratings.repartition(col('user_id'))

    stdout.write('DEBUG: Building all touched dictionary ...')
    all_touched = ratings.groupby('user_id').agg(collect_list('product_id').alias('all_touched_product_id'))
    all_touched_dict = all_touched.rdd.map(lambda row: row.asDict()).collect()
    all_touched_dict = dict([(elem['user_id'], elem['all_touched_product_id']) for elem in all_touched_dict])
    broadcasted_touched_dict = spark_session.sparkContext.broadcast(all_touched_dict)
    average_touched_items = np.mean([len(elems) for user, elems in all_touched_dict.items()])
    stdout.write('EVALUATION: Average touched items (non-unique) by user is ' + str(average_touched_items) + '\n')

    # get windows of touched items
    ratings = ratings.withColumn(
        'liked_product_id', collect_list(when(col('rating') > 3.0, col('product_id')).otherwise(lit(None))).over(user_window_preceding)
    )
    ratings = ratings.withColumn(
        'disliked_product_id', collect_list(when(col('rating') < 3.0, col('product_id')).otherwise(lit(None))).over(user_window_preceding)
    )
    ratings = ratings.withColumn('touched_product_id', collect_list(col('product_id')).over(user_window_preceding))

    # construct holdout set
    stdout.write('Constructing holdout set ...')
    ratings = ratings.withColumn('rank', row_number().over(user_window_present))
    holdout_thres = 10
    holdout_ratings = ratings.filter(col('rank') >= holdout_thres).\
        drop('rank').\
        drop('timestamp')
    prediction_states = holdout_ratings.filter(col('rank') == holdout_thres).select(
        col('user_id'),
        col('touched_product_id'),
        col('liked_product_id'),
        col('disliked_product_id')
    )
    final_states = holdout_ratings.groupby('user_id').agg(collect_set('product_id').alias('holdout_product_id'))
    holdout_frame = prediction_states.join(final_states, ['user_id'])

    holdout_types = dict([(field.name, str(field.dataType)) for field in holdout_frame.schema.fields])
    save_pickle(holdout_types, os.path.join(model_path, 'holdout_types.pkl'))
    cg_storage.save_file_gcs('holdout_types.pkl')

    holdout_frame = holdout_frame.toPandas()
    holdout_frame.to_csv(os.path.join(model_path, 'holdout.csv'), index=False)
    cg_storage.save_file_gcs('holdout.csv')

    ratings = ratings.filter(col('rank') < holdout_thres).\
        drop('rank').\
        drop('timestamp')
    ratings.persist()

    # negative sample
    stdout.write('DEBUG: Beginning negative sampling ... \n')
    num_products = int(max_product_id)
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

    # stratified split into training and validation
    stdout.write('DEBUG: Beginning stratified split ...')
    ratings_all = ratings_all.select('user_id', 'product_id', 'touched_product_id',
                                     'liked_product_id', 'disliked_product_id', 'target')
    train_df, val_df = stratified_split_distributed(ratings_all, 'target', spark_session)
    duration = time.time() - start_time
    stdout.write('BENCHMARKING: Runtime was ' + str(duration) + '\n')

    stdout.write('DEBUG: Converting dataframes to pandas ...' + '\n')
    train_pd = train_df.toPandas()
    train_pd.to_csv(os.path.join(model_path, 'train.csv'), index=False)
    cg_storage.save_file_gcs('train.csv')

    val_pd = val_df.toPandas()
    val_pd.to_csv(os.path.join(model_path, 'validation.csv'), index=False)
    cg_storage.save_file_gcs('validation.csv')

    stdout.write('DEBUG: Saving feature indices ... \n')
    feature_indices = dict([(feature, i) for i, feature in enumerate(ratings_all.schema.fieldNames())])
    save_pickle(feature_indices, os.path.join(model_path, 'feature_indices.pkl'))
    cg_storage.save_file_gcs('feature_indices.pkl')

    stdout.write('DEBUG: Saving feature types ... \n')
    feature_types = dict([(field.name, str(field.dataType)) for field in ratings_all.schema.fields])
    save_pickle(feature_types, os.path.join(model_path, 'feature_types.pkl'))
    cg_storage.save_file_gcs('feature_types.pkl')

if __name__ == '__main__':
    main()