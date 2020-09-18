from pyspark.sql.functions import *
from pyspark.sql.types import *

from pipeline.numpy_utils import *


def stratified_split_distributed(df, split_col, spark_session, train_ratio=0.8):
    """stratified split using spark"""
    split_col_index = df.schema.fieldNames().index(split_col)
    fractions = df.rdd.map(lambda x: x[split_col_index]).distinct().map(lambda x: (x, train_ratio)).collectAsMap()
    kb = df.rdd.keyBy(lambda x: x[split_col_index])
    train_rdd = kb.sampleByKey(False, fractions).map(lambda x: x[1])
    train_df = spark_session.createDataFrame(train_rdd, df.schema)
    val_df = df.exceptAll(train_df)
    return train_df, val_df


def negative_sampling_distributed(touched_col, item_col, num_items, sample_size):
    """perform negative sampling in parallel"""
    negative_ids = negative_sampling(
        np.concatenate([np.array(touched_col), np.array([item_col])], axis=0).tolist(),
        num_items, sample_size
    )
    return negative_ids.tolist()


def get_last_n_elements(arr, n):
    """used to get last n touched items"""
    return arr[-n:]










