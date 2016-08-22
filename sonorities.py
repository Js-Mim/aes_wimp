# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

"""     A short 'how to' use the trained models to separate solo and accompaning
        music sources from jazz recordings and remix them back.
"""

from TFMethods import TimeFrequencyDecomposition as TF
from MaskingMethods import FrequencyMasking as fm
import scipy.signal as sig
import numpy as np
import cPickle as pickle
import IOMethods as IO
import os, sys

eps = np.finfo(np.float32).tiny

def pan_gain_env(x, degrees, totalgain):
    """ Panoramic gain estimation for each independent channel and interpolation
        based on the input signal length.
        Args:
            x             : (array) Time domain input signal
            degrees       : (float) Desired panoramic/angle position
            totalgain     : (float) Linear gain scalar
        Returns:
            Lg            : (array) Time domain gain envelope for the left channel
            Rg            : (array) Time domain gain envelope for the right channel
    """
    angle = degrees * np.pi/180.

    Lg = np.ones(len(x), dtype = np.float32) * np.sqrt(2.)/2. * (np.cos(angle) - np.sin(angle))
    Rg = np.ones(len(x), dtype = np.float32) * np.sqrt(2.)/2. * (np.cos(angle) + np.sin(angle))

    return Lg * totalgain, Rg * totalgain

def val2vec(degree, gain):
    """ One hot encoding of the input location and gain values.
        Args:
            degrees       : (float) Desired panoramic/angle position
            gain          : (float) Linear gain scalar
        Returns:
            vec           : (array) A 40 elements array containing the binary representation
                                    of the desired values
    """
    lookupdeg = np.arange(-45, +50, 5)
    lookupgain = np.arange(0., 2.1, 0.1)
    vec = np.zeros((40,1), dtype = np.float32)

    # Degree check
    if degree < 0:
        diff = np.abs(np.abs(degree) - np.abs(lookupdeg[:-9]))
        loc = np.argmin(diff)

    if degree > 0:
        diff = np.abs(degree - lookupdeg[10:])
        loc = np.argmin(diff) + 10
    elif degree == 0:
        loc = 9


    # Gain Check
    diff = np.abs(gain - lookupgain)
    locB = np.argmin(diff) + len(lookupdeg)

    vec[loc, 0] = 1.
    vec[locB, 0] = 1.

    return vec

def vec2val(vec):
    """ One hot decoding of the input encoded vector.
        Args:
            vec           : (array) A 40 elements array containing the binary representation
                                    of the desired values
        Returns:
            degrees       : (float) Desired panoramic/angle position
            gain          : (float) Linear gain scalar

    """
    lookupdeg = np.arange(-45, +50, 5)
    lookupgain = np.arange(0., 2.1, 0.1)
    full_valvec = np.hstack((lookupdeg, lookupgain))
    locA, locB = np.where(vec == 1.)

    degrees = full_valvec[locA[0]]
    gain = full_valvec[locA[1]]

    return degrees, gain

def softmax(x):
    """ A simple softmax function """
    exps = np.sum(np.exp(x), axis=1)
    exps.shape = (exps.shape[0], 1)
    return np.exp(x) / exps

def sigmoid(x):
    """ A simple softmax function """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ A simple ReLU function """
    locx = np.where(x < 0.)
    x[locx[0], locx[1]] = 0.
    return x

def remix_solo(x):
    """ The core method to analyse, separate, estimate, remix and reconstruct audio mixtures and sources.
        Args:
            x             : (2D ndarray) The two-channel mixture time domain waveform
        Returns:
            x             : (2D ndarray) The two-channel mixture time domain waveform
            yhat          : (array)      Single channel solo instrument time domain waveform
            yhatb         : (2D ndarray) The two-channel accompanying instruments time domain waveform
            ymix          : (2D ndarray) The two-channel remixed time domain waveform
    """
    # Load models using pickle
    print('Loading models')

    # Check for os, to avoid some windows crushes
    plat = sys.platform
    if plat  == 'linux' or plat == 'linux2' or plat == 'darwin' :
        ww = pickle.load(open('solo_suppression_mag.p', 'rb'))
        wwpan = pickle.load(open('pannet_mag.p', 'rb'))
    else :
        fileA = open('solo_suppression_mag.p', 'rb')
        ww = pickle.load(fileA,encoding='latin1')
        fileB = open('pannet_mag.p', 'rb')
        wwpan = pickle.load(fileB,encoding='latin1')
        del fileA, fileB

    hop = 512
    N = 4096
    wsz = 2049
    # Left/Right/Mid Analysis
    xL = x[:, 0]
    xR = x[:, 1]
    MmX, MpX = TF.STFT((xL+xR) * 0.5, sig.bartlett(wsz, True), N, hop)
    LmX, LpX = TF.STFT(xL, sig.bartlett(wsz, True), N, hop)
    RmX, RpX = TF.STFT(xR, sig.bartlett(wsz, True), N, hop)

    print('Extracting Solo Information')
    ### Hidden Layer Representation 1
    Trs = sigmoid(np.dot(MmX, ww[2]) + ww[3])
    act = relu(np.dot(MmX, ww[0]) + ww[1])
    act *= Trs
    hl = act + (1. - Trs) * MmX

    ### Hidden Layer Representation 2
    Trs = sigmoid(np.dot(hl, ww[6]) + ww[7])
    act = relu(np.dot(hl, ww[4]) + ww[5])
    act *= Trs
    hl = act + (1. - Trs) * hl

    ### Hidden Layer Representation 3
    Trs = sigmoid(np.dot(hl, ww[10]) + ww[11])
    act = relu(np.dot(hl, ww[8]) + ww[9])
    act *= Trs
    hl = act + (1. - Trs) * hl

    ### Hidden Layer Representation 4
    Trs = sigmoid(np.dot(hl, ww[14]) + ww[15])
    act = relu(np.dot(hl, ww[12]) + ww[13])
    act *= Trs
    hl = act + (1. - Trs) * hl

    ### Output Layer
    Trs = sigmoid(np.dot(hl, ww[18]) + ww[19])
    act = relu(np.dot(hl, ww[16]) + ww[17])
    act *= Trs
    hl = ((act + (1. - Trs) * hl) + eps)

    # Monophonic Solo
    yhat = TF.iSTFT(hl, MpX, wsz, hop)

    # Stereo instrumentation
    print('Estimating accompaniment instrumentation')
    mask = fm(LmX, hl, [(LmX-hl).clip(0.)], [], [], alpha = 1.3, method = 'alphaWiener')
    mshatL = mask(reverse = True)

    mask = fm(RmX, hl, [(RmX-hl).clip(0.)], [], [], alpha = 1.3, method = 'alphaWiener')
    mshatR = mask(reverse = True)

    # Time-domain reconstruction
    yhatbL = TF.iSTFT(mshatL, LpX, wsz, hop)
    yhatbR = TF.iSTFT(mshatR, RpX, wsz, hop)

    yhatb = np.vstack((yhatbL, yhatbR)).T

    # Mixing coefficients Estimation
    print('Estimating Mixing Coefficients')
    ### Hidden Layer Representation 1
    Trs = sigmoid(np.dot(hl, wwpan[2]) + wwpan[3])
    act = relu(np.dot(hl, wwpan[0]) + wwpan[1])
    act *= Trs
    hl = act + (1. - Trs) * hl

    ### Hidden Layer Representation 2
    Trs = sigmoid(np.dot(hl, wwpan[6]) + wwpan[7])
    act = relu(np.dot(hl, wwpan[4]) + wwpan[5])
    act *= Trs
    hl = act + (1. - Trs) * hl

    mix_vec = softmax(np.dot(hl, wwpan[8]) + wwpan[9])
    mix_vec = np.sum(mix_vec, axis=0)

    # Acquiring locations
    degloc = np.argmax(mix_vec[19:])
    gloc = np.argmax(mix_vec[:19])
    mix_vec = np.zeros((40,1), dtype = np.float32)

    mix_vec[degloc + 19] = 1.
    mix_vec[gloc] = 1.

    print('Performing Mixing')
    degrees, gain = vec2val(mix_vec)
    LGenv, RGenv = pan_gain_env(yhat, degrees, gain)

    ymix = np.vstack((yhat * LGenv, yhat * RGenv)).T + yhatb

    return x, yhat[:x.shape[0]], yhatb[:x.shape[0], :], ymix[:x.shape[0], :]

if __name__ == '__main__':
    # Sanity check for the existence of the models
    if os.path.exists('solo_suppression_mag.p') and os.path.exists('pannet_mag.p'):
        print('Models Located!')
    else :
        raise IOError('Trained Models Not Found! Please refer to README file!')

    # Path for audio files
    loadpath = 'wav/'
    savepath = 'wav/'
    filelist = sorted(os.listdir(loadpath))

    # Iterate over the list of files
    for indx in filelist:
    	if not indx.startswith('.'):
	        xfilename = os.path.join(loadpath, indx)
	        # Reading
	        x, fs = IO.AudioIO.wavRead(xfilename, mono = False)
	        # Check for clipped audio data
	        if np.max(np.abs(x)) >= 0.99 :
	            print('Clipping')
	            locx = np.where(np.abs(x) >= 0.93)
	            x[locx[0], locx[1]] = 0.93 * np.sign(x[locx[0], locx[1]])

	        # Feed the system
	        xa, yhat, yhatb, ymix = remix_solo(x)

	        # Original Audio file
	        #IO.AudioIO.sound(xa, fs)
	        # Estimated solo instrumet
	        IO.AudioIO.wavWrite(yhat, fs, 16, xfilename[:-4]+'_solo.wav')
	        # Estimated accompaniment music
	        IO.AudioIO.wavWrite(yhatb, fs, 16, xfilename[:-4]+'_acc.wav')
	        # Automatic Mixture
	        IO.AudioIO.wavWrite(ymix, fs, 16, xfilename[:-4]+'_remixed.wav')