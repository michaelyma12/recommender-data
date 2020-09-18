import os
from pickle import dump, load


def save_pickle(elem, save_path):
    """Save an item with pickle"""
    pickle_uni = open(save_path, 'wb')
    dump(elem, pickle_uni)
    pickle_uni.close()


def handle_path(path):
    """create ticker dir if missing"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path
