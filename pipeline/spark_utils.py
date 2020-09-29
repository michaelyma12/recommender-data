from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer

from pipeline.numpy_utils import *
from pipeline.file_utils import *


def stratified_split_distributed(df, split_col, spark_session, train_ratio=0.8):
    """stratified split using spark"""
    split_col_index = df.schema.fieldNames().index(split_col)
    fractions = df.rdd.map(lambda x: x[split_col_index]).distinct().map(lambda x: (x, train_ratio)).collectAsMap()
    kb = df.rdd.keyBy(lambda x: x[split_col_index])
    train_rdd = kb.sampleByKey(False, fractions).map(lambda x: x[1])
    train_df = spark_session.createDataFrame(train_rdd, df.schema)
    val_df = df.exceptAll(train_df)
    return train_df, val_df


def negative_sampling_distributed_rolling(touched_col, item_col, num_items, sample_size):
    """perform negative sampling in parallel on a rolling basis"""
    negative_ids = negative_sampling(
        np.concatenate([np.array(touched_col), np.array([item_col])], axis=0).tolist(),
        num_items, sample_size
    )
    return negative_ids.tolist()


def negative_sampling_distributed_f(broadcasted_touched_dictionary):
    """perform negative sampling in parallel using all of a user's touched products"""
    def f(user_col, num_items, sample_size):
        return negative_sampling(broadcasted_touched_dictionary.value.get(user_col),
                                 num_items, sample_size).tolist()
    return f


def get_last_n_elements(arr, n):
    """used to get last n touched items"""
    return arr[-n:]


def encode_column(data, column):
    """encode column and save mappings"""
    column_encoder = StringIndexer().setInputCol(column).setOutputCol('encoded_{}'.format(column))
    encoder_model = column_encoder.fit(data)
    data = encoder_model.transform(data).withColumn('encoded_{}'.format(column),
                                                    col('encoded_{}'.format(column)).cast('int'))
    data = data.drop(column)
    data = data.withColumnRenamed('encoded_{}'.format(column), column)
    data = data.withColumn(column, col(column) + lit(1))
    id_mapping = dict([(elem, i + 1) for i, elem in enumerate(encoder_model.labels)])
    return data, id_mapping










