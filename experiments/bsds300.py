import h5py


def load_bsds300_dataset(rand_state):
    # Load the dataset
    file = h5py.File('datasets/BSDS300/BSDS300.hdf5', 'r')
    data_train = file['train'][:]
    data_val = file['validation'][:]
    data_test = file['test'][:]
    data_train = data_train.astype('float32')
    data_val = data_val.astype('float32')
    data_test = data_test.astype('float32')
    return data_train, data_val, data_test
