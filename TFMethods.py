# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import math
import numpy as np
from scipy.fftpack import fft, ifft, dct, dst
from scipy.signal import firwin2, freqz, cosine, hanning
from scipy.interpolate import InterpolatedUnivariateSpline as uspline

eps = np.finfo(np.float32).tiny

class TimeFrequencyDecomposition:
    """ A Class that performs time-frequency decompositions by means of a
        Discrete Fourier Transform, using Fast Fourier Transform algorithm
        by SciPy. Currently, inverse synthesis is being supported for
        Gabor transformations and it's variants, alongside with Zero-phase windowing
        technique with arbitrary window size (odd numbers are prefered).
    """
    @staticmethod
    def DFT(x, w, N):
        """ Discrete Fourier Transformation(Analysis) of a given real input signal
        via an FFT implementation from scipy. Single channel is being supported.
        Args:
            x       : (array) Real time domain input signal
            w       : (array) Desired windowing function
            N       : (int)   FFT size
        Returns:
            magX    : (2D ndarray) Magnitude Spectrum
            phsX    : (2D ndarray) Phase Spectrum
        """

        # Half spectrum size containing DC component
        hlfN = (N/2)+1

        # Half window size. Two parameters to perform zero-phase windowing technique
        hw1 = int(math.floor((w.size+1)/2))
        hw2 = int(math.floor(w.size/2))

        # Window the input signal
        winx = x*w

        # Initialize FFT buffer with zeros and perform zero-phase windowing
        fftbuffer = np.zeros(N)
        fftbuffer[:hw1] = winx[hw2:]
        fftbuffer[-hw2:] = winx[:hw2]

        # Compute DFT via scipy's FFT implementation
        X = fft(fftbuffer)

        # Acquire magnitude and phase spectrum
        magX = (np.abs(X[:hlfN]))
        phsX = (np.angle(X[:hlfN]))

        return magX, phsX

    @staticmethod
    def iDFT(magX, phsX, wsz):
        """ Discrete Fourier Transformation(Synthesis) of a given spectral analysis
        via an inverse FFT implementation from scipy.
        Args:
            magX    : (2D ndarray) Magnitude Spectrum
            phsX    : (2D ndarray) Phase Spectrum
            wsz     :  (int)   Synthesis window size
        Returns:
            y       : (array) Real time domain output signal
        """

        # Get FFT Size
        hlfN = magX.size;
        N = (hlfN-1)*2

        # Half of window size parameters
        hw1 = int(math.floor((wsz+1)/2))
        hw2 = int(math.floor(wsz/2))

        # Initialise synthesis buffer with zeros
        fftbuffer = np.zeros(N)
        # Initialise output spectrum with zeros
        Y = np.zeros(N, dtype = complex)
        # Initialise output array with zeros
        y = np.zeros(wsz)

        # Compute complex spectrum(both sides) in two steps
        Y[0:hlfN] = magX * np.exp(1j*phsX)
        Y[hlfN:] = magX[-2:0:-1] * np.exp(-1j*phsX[-2:0:-1])

        # Perform the iDFT
        fftbuffer = np.real(ifft(Y))

        # Roll-back the zero-phase windowing technique
        y[:hw2] = fftbuffer[-hw2:]
        y[hw2:] = fftbuffer[:hw1]

        return y

    @staticmethod
    def STFT(x, w, N, hop):
        """ Short Time Fourier Transform analysis of a given real input signal,
        via the above DFT method.
        Args:
            x   : 	(array)  Magnitude Spectrum
            w   :   (array)  Desired windowing function
            N   :   (int)    FFT size
            hop :   (int)    Hop size
        Returns:
            sMx :   (2D ndarray) Stacked arrays of magnitude spectra
            sPx :   (2D ndarray) Stacked arrays of phase spectra
        """

        # Analysis Parameters
        wsz = w.size

        hw1 = int(math.floor((wsz+1)/2))
        hw2 = int(math.floor(wsz/2))

        # Add some zeros at the start and end of the signal to avoid window smearing
        x = np.append(np.zeros(3*hop),x)
        x = np.append(x, np.zeros(3*hop))

        # Initialize sound pointers
        pin = 0
        pend = x.size - wsz
        indx = 0

        # Normalise windowing function
        w = w / sum(w)

        # Initialize storing matrix
        xmX = np.zeros((len(x)/hop, N/2 + 1), dtype = np.float32)
        xpX = np.zeros((len(x)/hop, N/2 + 1), dtype = np.float32)

        # Analysis Loop
        while pin <= pend:
            # Acquire Segment
            xSeg = x[pin:pin+wsz]

            # Perform DFT on segment
            mcX, pcX = TimeFrequencyDecomposition.DFT(xSeg, w, N)

            xmX[indx, :] = mcX
            xpX[indx, :] = pcX

            # Update pointers and indices
            pin += hop
            indx += 1

        return xmX, xpX

    @staticmethod
    def iSTFT(xmX, xpX, wsz, hop) :
        """ Short Time Fourier Transform synthesis of given magnitude and phase spectra,
        via the above iDFT method.
        Args:
            xmX :   (2D ndarray)  Magnitude Spectrum
            xpX :   (2D ndarray)  Phase Spectrum
            wsz :   (int)    Synthesis Window size
            hop :   (int)    Hop size
        Returns :
            y   :   (array) Synthesised time-domain real signal.
        """

        # Acquire half window sizes
        hw1 = int(math.floor((wsz+1)/2))
        hw2 = int(math.floor(wsz/2))

        # Acquire the number of STFT frames
        numFr = xmX.shape[0]

        # Initialise output array with zeros
        y = np.zeros(numFr * hop + hw1 + hw2)

        # Initialise sound pointer
        pin = 0

        # Main Synthesis Loop
        for indx in range(numFr):
            # Inverse Discrete Fourier Transform
            ybuffer = TimeFrequencyDecomposition.iDFT(xmX[indx, :], xpX[indx, :], wsz)

            # Overlap and Add
            y[pin:pin+wsz] += ybuffer*hop

            # Advance pointer
            pin += hop

        # Delete the extra zeros that the analysis had placed
        y = np.delete(y, range(3*hop))
        y = np.delete(y, range(y.size-(3*hop + 1), y.size))

        return y

    @staticmethod
    def nuttall4b(M, sym=False):
        """
        Returns a minimum 4-term Blackman-Harris window according to Nuttall.
        The typical Blackman window famlity define via "alpha" is continuous
        with continuous derivative at the edge. This will cause some errors
        to short time analysis, using odd length windows.

        Args    :
            M   :   (int)   Number of points in the output window.
            sym :   (array) Synthesised time-domain real signal.

        Returns :
            w   :   (ndarray) The windowing function

        References :
            [1] Heinzel, G.; Rüdiger, A.; Schilling, R. (2002). Spectrum and spectral density
               estimation by the Discrete Fourier transform (DFT), including a comprehensive
               list of window functions and some new flat-top windows (Technical report).
               Max Planck Institute (MPI) für Gravitationsphysik / Laser Interferometry &
               Gravitational Wave Astronomy, 395068.0

            [2] Nuttall A.H. (1981). Some windows with very good sidelobe behaviour. IEEE
               Transactions on Acoustics, Speech and Signal Processing, Vol. ASSP-29(1):
               84-91.
        """

        if M < 1:
            return np.array([])
        if M == 1:
            return np.ones(1, 'd')
        if not sym :
            M = M + 1

        a = [0.355768, 0.487396, 0.144232, 0.012604]
        n = np.arange(0, M)
        fac = n * 2 * np.pi / (M - 1.0)

        w = (a[0] - a[1] * np.cos(fac) +
             a[2] * np.cos(2 * fac) - a[3] * np.cos(3 * fac))

        if not sym:
            w = w[:-1]

        return w

    @staticmethod
    def coreModulation(win, N):
        """
            Method to produce Analysis and Synthesis matrices for the offline
            complex PQMF class.

            Arguments  :
                win    :  (1D Array) Windowing function
                N      :  (int) Number of subbands

            Returns  :
                Cos   :   (2D Array) Cosine Modulated Polyphase Matrix
                Sin   :   (2D Array) Sine Modulated Polyphase Matrix


            Usage  : Cos, Sin = TimeFrequencyDecomposition.coreModulation(qmfwin, N)
        """

        lfb = len(win)
        # Initialize Storing Variables
        Cos = np.zeros((N,lfb), dtype = np.float32)
        Sin = np.zeros((N,lfb), dtype = np.float32)

        # Generate Matrices
        for k in xrange(0, lfb):
            for n in xrange(0, N):
                Cos[n, k] = win[k] * np.cos(np.pi/N * (n + 0.5) * (k + 0.5 + N/2)) * np.sqrt(2. / N)
                Sin[n, k] = win[k] * np.sin(np.pi/N * (n + 0.5) * (k + 0.5 + N/2)) * np.sqrt(2. / N)

        return Cos, Sin

    @staticmethod
    def complex_analysis(x, N = 1024):
        """
            Method to compute the subband samples from time domain signal x.
            A complex output matrix will be computed using DCT and DST.

            Arguments   :
                x       : (1D Array) Input signal
                N       : (int)      Number of sub-bands

            Returns     :
                y       : (2D Array) Complex output of QMF analysis matrix (Cosine)

            Usage       : y = TimeFrequencyDecomposition.complex_analysis(x, N)

        """
        # Parameters and windowing function design
        win = cosine(2*N, True)
        win /= np.sum(win)
        lfb = len(win)
        nTimeSlots = len(x)/N - 2

        # Initialization
        ycos = np.zeros((len(x)/N, N), dtype = np.float32)
        ysin = np.zeros((len(x)/N, N), dtype = np.float32)

        # Analysis Matrices
        Cos, Sin = TimeFrequencyDecomposition.coreModulation(win, N)

        # Perform Complex Analysis
        for m in xrange(0, nTimeSlots):
            ycos[m, :] = np.dot(x[m * N : m * N + lfb], Cos.T)
            ysin[m, :] = np.dot(x[m * N : m * N + lfb], Sin.T)

        y = ycos + 1j *  ysin

        return y

    @staticmethod
    def complex_synthesis(y, N = 1024):
        """
            Method to compute the resynthesis of the MDCST.
            A complex input matrix is asummed as input, derived from DCT and DST.

            Arguments   :
                y       : (2D Array) Complex Representation

            Returns     :
                xrec    : (1D Array) Time domain reconstruction

            Usage       : xrec = TimeFrequencyDecomposition.complex_synthesis(y, N)

        """
        # Parameters and windowing function design
        win = cosine(2*N, True)
        win *= np.sum(win)
        lfb = len(win)
        nTimeSlots = y.shape[0]
        SignalLength = nTimeSlots * N + 2 * N

        # Synthesis matrices
        Cos, Sin = TimeFrequencyDecomposition.coreModulation(win, N)

        # Initialization
        zcos = np.zeros((1, SignalLength), dtype = np.float32)
        zsin = np.zeros((1, SignalLength), dtype = np.float32)

        # Perform Complex Synthesis
        for m in xrange(0, nTimeSlots):
            zcos[0, m * N : m * N + lfb] += np.dot(np.real(y[m, :]).T, Cos)
            zsin[0, m * N : m * N + lfb] += np.dot(np.imag(y[m, :]).T, Sin)

        xrec = 0.5 * (zcos + zsin)

        return xrec.T

    @staticmethod
    def real_synthesis(y, N = 1024):
        """
            Method to compute the resynthesis of the MDCT.
            A complex input matrix is asummed as input, derived from DCT typeIV.

            Arguments   :
                y       : (2D Array) Complex Representation

            Returns     :
                xrec    : (1D Array) Time domain reconstruction

            Usage       : xrec = TimeFrequencyDecomposition.complex_synthesis(y, N)

        """
        # Parameters and windowing function design
        win = cosine(2*N, True)
        win *= np.sum(win)
        lfb = len(win)
        nTimeSlots = y.shape[0]
        SignalLength = nTimeSlots * N + 2 * N

        # Synthesis matrices
        Cos, _ = TimeFrequencyDecomposition.coreModulation(win, N)

        # Initialization
        zcos = np.zeros((1, SignalLength), dtype = np.float32)

        # Perform Complex Synthesis
        for m in xrange(0, nTimeSlots):
            zcos[0, m * N : m * N + lfb] += np.dot((y[m, :]).T, Cos)

        xrec = zcos
        return xrec.T

class CepstralDecomposition:
    """ A Class that performs a cepstral decomposition based on the
        logarithmic observed magnitude spectrogram. As appears in:
        "A Novel Cepstral Representation for Timbre Modelling of
        Sound Sources in Polyphonic Mixtures", Z.Duan, B.Pardo, L. Daudet.
    """
    @staticmethod
    def computeUDCcoefficients(freqPoints = 2049, points = 2049, fs = 44100, melOption = False):
        """ Computation of M matrix that contains the coefficients for
        cepstral modelling architecture.
        Args:
            freqPoints   :     (int)  Number of frequencies to model
            points       : 	   (int)  The cepstum order (number of coefficients)
            fs           :     (int)  Sampling frequency
            melOption    :     (bool) Compute Mel-uniform discrete cepstrum
        Returns:
            M            :     (ndarray) Matrix containing the coefficients
        """
        M = np.empty((freqPoints, points), dtype = np.float32)
        # Array obtained by the order number of cepstrum
        p = np.arange(points)
        # Array with frequncy bin indices
        f = np.arange(freqPoints)

        if (freqPoints % 2 == 0):
            fftsize = (freqPoints)*2
        else :
            fftsize = (freqPoints-1)*2

        # Creation of frequencies from frequency bins
        farray = f * float(fs) / fftsize
        if (melOption):
            melf = 2595.0 * np.log10(1.0 + farray * fs/700.0)
            melHsr = 2595.0 * np.log10(1.0 + (fs/2.0) * fs/700.0)
            farray = (0.5 * melf) / (melHsr)
        else:
            farray = farray/(fs)

        twoSqrt = np.sqrt(2.0)

        for indx in range(M.shape[0]):
            M[indx, :] = (np.cos(2.0 * np.pi * p * farray[indx]))
            M[indx, 1:] *= twoSqrt

        return M

if __name__ == "__main__":
    import IOMethods as IO
    import matplotlib.pyplot as plt
    np.random.seed(218)
    # Test
    #kSin = np.cos(np.arange(88200) * (1000.0 * (3.1415926 * 2.0) / 44100)) * 0.5
    mix, fs = IO.AudioIO.audioRead('testFiles/cmente.mp3', mono = True)
    mix = mix[44100*25:44100*25 + 882000] * 0.25
    noise = np.random.uniform(-30., 30., len(mix))

    # STFT/iSTFT Test
    w = np.hanning(1025)
    magX, phsX =  TimeFrequencyDecomposition.STFT(mix, w, 2048, 512)
    magN, phsN =  TimeFrequencyDecomposition.STFT(noise, w, 2048, 512)
