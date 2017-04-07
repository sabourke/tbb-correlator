#!/usr/bin/env python

import numpy as np
import h5py
import multiprocessing
import sys
import copy_reg
import types
import argparse

DEFAULT_FFT_SIZE = 1024
READ_FFTS = 8 # Read this amount of data from file at a time

def num_streams(h5file):
    with h5py.File(h5file, "r") as h5:
        return len(h5.values()[0])

def data_size(h5file):
    """Return the length of the shortest stream"""
    with h5py.File(h5file, "r") as h5:
        return min([rcu.size for rcu in h5.values()[0].values()])

def stream_attr(h5file, attr):
    """Return the specified attribute for all streams"""
    with h5py.File(h5file, "r") as h5:
        return np.array([rcu.attrs[attr][0] for rcu in h5.values()[0].values()])

def sample_rate(h5file):
    """Return the sample rate of the data streams"""
    sample_rates = stream_attr(h5file, "SAMPLE_FREQUENCY_VALUE")
    sr_unit = stream_attr(h5file, "SAMPLE_FREQUENCY_UNIT")
    assert (sr_unit == "MHz").all()
    assert (sample_rates == sample_rates[0]).all()
    return sample_rates[0] * 1e6

def read_data(h5file, rcu, start, stop):
    with h5py.File(h5file, "r") as h5:
        data_node = h5.values()[0].values()[rcu]
        if stop > data_node.size:
            # h5 slicing silently returns less data if you go past the end
            raise IndexError, "%d > node size: %d" % (stop, data_node.size)
        return data_node[start:stop]

def correlate(arr_in):
    f = np.fft.rfft(arr_in)
    return f[:,np.newaxis,:] * f[np.newaxis,:,:].conj()

class correlation_job(object):
    """Class to encapsulate a correlation job to facilitate parallelisation."""

    def __init__(self, filename, n_inputs, delays, fft_size=DEFAULT_FFT_SIZE, offset=0):
        self.filename = filename
        self.n_inputs = n_inputs
        self.fft_size = fft_size
        self.n_freq = self.fft_size / 2 + 1
        self.window = np.hanning(self.fft_size)
        self.delays = delays
        self.offset = offset

    def zero_out_array(self):
        return np.zeros((self.n_inputs, self.n_inputs, self.n_freq), dtype=np.complex64)

    def correlate_chunk(self, n):
        print "Correlating chunk", n
        arr_in = np.empty(shape=(self.n_inputs, READ_FFTS * self.fft_size), dtype=np.float32)
        for i in range(self.n_inputs):
            start = n * READ_FFTS * self.fft_size + self.delays[i] + self.offset
            stop = (n + 1) * READ_FFTS * self.fft_size + self.delays[i] + self.offset
            arr_in[i] = read_data(self.filename, i, start, stop)
        tmp_out = self.zero_out_array()
        for i in range(READ_FFTS):
            tmp_in = arr_in[:,i*self.fft_size:(i+1)*self.fft_size] * self.window
            tmp_out += (correlate(tmp_in) - tmp_out) / (i + 1)
        return tmp_out

# functions to allow instance methods to be pickeled so they
# can be used with the multiprocessing module
# ref: https://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", type=float, help="Integration time", default=0.1)
    parser.add_argument("-s", "--offset", type=int, help="Offset in samples to skip at start", default=0)
    parser.add_argument("-i", "--integration", type=int, help="Integration to correlate", default=0)
    parser.add_argument("-o", "--outname", type=str, help="Output file name", default="tbb_xc.npy")
    parser.add_argument("-f", "--fftsize", type=int, help="Size of FFT. Out chans will be fft//2+1", default=DEFAULT_FFT_SIZE)
    parser.add_argument("-p", "--processes", type=int, help="Number of correlation processes to use in parallel. Default is 1 per core", default=None)
    parser.add_argument("infile", type=str, help="Input HDF5 file")
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    delays = stream_attr(args.infile, "SAMPLE_NUMBER") * -1
    delays -= delays.min()
    n_input = num_streams(args.infile)
    samp_rate = sample_rate(args.infile)
    corr_job = correlation_job(args.infile, n_input, delays, args.fftsize, args.offset)
    start_chunk = int(args.integration * args.time * samp_rate / READ_FFTS / args.fftsize)
    finish_chunk = int((args.integration + 1) * args.time * samp_rate / READ_FFTS / args.fftsize)
    print "Correlating chunks {} to {}".format(start_chunk, finish_chunk)

    out_data = corr_job.zero_out_array()
    p = multiprocessing.Pool(args.processes)
    for i, result in enumerate(p.imap(corr_job.correlate_chunk, xrange(start_chunk, finish_chunk))):
        out_data += (result - out_data) / (i + 1)

    np.save(args.outname, out_data)

if __name__ == "__main__":
    main()
