import gzip
import pickle as p


def save_zipped_pickle(obj, filename, protocol=-1):
    print('saving data to = {0}'.format(filename))
    with gzip.open(filename, 'wb') as f:
        p.dump(obj, f, protocol)


def load_zipped_pickle(filename):
    #print('loading cached data from = {0}'.format(filename))
    with gzip.open(filename, 'rb') as f:
        loaded_object = p.load(f)
        return loaded_object