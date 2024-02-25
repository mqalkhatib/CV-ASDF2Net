import numpy as np
from spectral.io import envi
import os.path
from pathlib import Path
import scipy.io as sio


def load_data(name):
    if name == 'FL_T':
        path = 'Datasets\Flevoland\T3'
        path = Path(path)
        first_read = envi.open(path / 'T11.bin.hdr', path / 'T11.bin').read_band(0)
        T = np.zeros(first_read.shape + (6,), dtype=np.complex64)
    
        T[:, :, 0] = first_read
        T[:, :, 1] = envi.open(path / 'T22.bin.hdr', path / 'T22.bin').read_band(0)
        T[:, :, 2] = envi.open(path / 'T33.bin.hdr', path / 'T33.bin').read_band(0)
        T[:, :, 3] = envi.open(path / 'T12_real.bin.hdr', path / 'T12_real.bin').read_band(0) + \
                1j * envi.open(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin').read_band(0)
        T[:, :, 4] = envi.open(path / 'T13_real.bin.hdr', path / 'T13_real.bin').read_band(0) + \
                1j * envi.open(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin').read_band(0)
        T[:, :, 5] = envi.open(path / 'T23_real.bin.hdr', path / 'T23_real.bin').read_band(0) + \
                1j * envi.open(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin').read_band(0)
                
        labels = sio.loadmat('Datasets\Flevoland\FlevoLand_gt.mat')['gt']
##############################################################################
   
    elif name == 'SF':
        path = 'Datasets\san_francisco\T3'
        path = Path(path)
        first_read = envi.open(path / 'T11.bin.hdr', path / 'T11.bin').read_band(0)
        T = np.zeros(first_read.shape + (6,), dtype=np.complex64)
        
        T[:, :, 0] = first_read
        T[:, :, 1] = envi.open(path / 'T22.bin.hdr', path / 'T22.bin').read_band(0)
        T[:, :, 2] = envi.open(path / 'T33.bin.hdr', path / 'T33.bin').read_band(0)
        T[:, :, 3] = envi.open(path / 'T12_real.bin.hdr', path / 'T12_real.bin').read_band(0) + \
                1j * envi.open(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin').read_band(0)
        T[:, :, 4] = envi.open(path / 'T13_real.bin.hdr', path / 'T13_real.bin').read_band(0) + \
                1j * envi.open(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin').read_band(0)
        T[:, :, 5] = envi.open(path / 'T23_real.bin.hdr', path / 'T23_real.bin').read_band(0) + \
                1j * envi.open(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin').read_band(0)
        
        labels = sio.loadmat('Datasets\san_francisco\SanFrancisco_gt.mat')['gt']              
         
##############################################################################
    elif name == 'ober':
        path = 'Datasets\Oberpfaffenhofen\ESAR_Oberpfaffenhofen_T6'
        path = Path(path)
        first_read = envi.open(path / 'T11.bin.hdr', path / 'T11.bin').read_band(0)
        T = np.zeros(first_read.shape + (21,), dtype=np.complex64)
    
        T[:, :, 0] = first_read
        T[:, :, 1] = envi.open(path / 'T22.bin.hdr', path / 'T22.bin').read_band(0)
        T[:, :, 2] = envi.open(path / 'T33.bin.hdr', path / 'T33.bin').read_band(0)
        T[:, :, 3] = envi.open(path / 'T44.bin.hdr', path / 'T44.bin').read_band(0)
        T[:, :, 4] = envi.open(path / 'T55.bin.hdr', path / 'T55.bin').read_band(0)
        T[:, :, 5] = envi.open(path / 'T66.bin.hdr', path / 'T66.bin').read_band(0)        
        T[:, :, 6] = envi.open(path / 'T12_real.bin.hdr', path / 'T12_real.bin').read_band(0) + \
                1j * envi.open(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin').read_band(0)
        T[:, :, 7] = envi.open(path / 'T13_real.bin.hdr', path / 'T13_real.bin').read_band(0) + \
                1j * envi.open(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin').read_band(0)
        T[:, :, 8] = envi.open(path / 'T14_real.bin.hdr', path / 'T14_real.bin').read_band(0) + \
                1j * envi.open(path / 'T14_imag.bin.hdr', path / 'T14_imag.bin').read_band(0)
        T[:, :, 9] = envi.open(path / 'T15_real.bin.hdr', path / 'T15_real.bin').read_band(0) + \
                1j * envi.open(path / 'T15_imag.bin.hdr', path / 'T15_imag.bin').read_band(0)
        T[:, :, 10] = envi.open(path / 'T16_real.bin.hdr', path / 'T16_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T16_imag.bin.hdr', path / 'T16_imag.bin').read_band(0)
        T[:, :, 11] = envi.open(path / 'T23_real.bin.hdr', path / 'T23_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin').read_band(0)
        T[:, :, 12] = envi.open(path / 'T24_real.bin.hdr', path / 'T24_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T24_imag.bin.hdr', path / 'T24_imag.bin').read_band(0)
        T[:, :, 13] = envi.open(path / 'T25_real.bin.hdr', path / 'T25_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T25_imag.bin.hdr', path / 'T25_imag.bin').read_band(0)
        T[:, :, 14] = envi.open(path / 'T26_real.bin.hdr', path / 'T26_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T26_imag.bin.hdr', path / 'T26_imag.bin').read_band(0)    
        T[:, :, 15] = envi.open(path / 'T34_real.bin.hdr', path / 'T34_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T34_imag.bin.hdr', path / 'T34_imag.bin').read_band(0)
        T[:, :, 16] = envi.open(path / 'T35_real.bin.hdr', path / 'T35_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T35_imag.bin.hdr', path / 'T35_imag.bin').read_band(0)
        T[:, :, 17] = envi.open(path / 'T36_real.bin.hdr', path / 'T36_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T36_imag.bin.hdr', path / 'T36_imag.bin').read_band(0)
        T[:, :, 18] = envi.open(path / 'T45_real.bin.hdr', path / 'T45_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T45_imag.bin.hdr', path / 'T45_imag.bin').read_band(0)
        T[:, :, 19] = envi.open(path / 'T46_real.bin.hdr', path / 'T46_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T46_imag.bin.hdr', path / 'T46_imag.bin').read_band(0)
        T[:, :, 20] = envi.open(path / 'T56_real.bin.hdr', path / 'T56_real.bin').read_band(0) + \
                 1j * envi.open(path / 'T56_imag.bin.hdr', path / 'T56_imag.bin').read_band(0)

        labels = sio.loadmat('Datasets\Oberpfaffenhofen\Oberpfaffenhofen_gt.mat')['gt']
##############################################################################
   
    else:
        print("Incorrect data name")
        
        
        
    return T, labels

