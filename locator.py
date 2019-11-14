import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.interpolate import interp1d
from numpy import linalg as la
from sympy.utilities.iterables import multiset_combinations
from polynomat import PolynoMat
from scipy.signal import fftconvolve
from multiprocessing import Pool
import scipy.linalg as la
from scipy.linalg import eigh
from scipy import signal
from numpy import dot, sqrt, argsort, abs
import pickle
import tqdm



# """
# This messy code is here because I couldn't get pyfftw installed on my Windows OS, but got it fine on my Linux OS.
# Essentially, I wanted the Locator to use pyfftw when it was available, or numpy's fft module otherwise.
# """
wf = False
pftw = False
try:
    raise ImportError
    import pyfftw.interfaces.numpy_fft as fft_pack
    import pyfftw

    pftw = True
    pyfftw.interfaces.cache.enable()
    try:
        with open('pyfftw_wisdom.txt', 'rb') as wizfile:
            pyfftw.import_wisdom(pickle.load(wizfile))
        wf = True
    except:
        pass
except ImportError as e:
    print(e)
    import numpy.fft as fft_pack


class Locator:
    """Used to locate the source of a sound in a multi-track .wav file.

    The locator class may also be used to generate su=imulated data by means of the shift_sound method, which
    takes in a mono wav file and a set of coordinates and produces a multi-track wav simulating the data which
    would have been recorded were the signal to have came from the provided location.
    """
    sample_rate: int = None
    """The sample data of the currently loaded data."""
    data: np.array = None
    """The current data, stored in a numpy array."""
    sound_speed: float = 343.1
    """The speed of sound in air to use in the calculations."""
    _hm_domain_ = None
    _radial_domain_ = None
    _cor_fns_ = {}
    _mic_pairs_ = None
    _GCC_proc_ = ""

    def __init__(self, mic_locations=((0.325, 0.000), (-0.160, 0.248), (-0.146, -0.258), (-0.001, 0.002)),
                 file_path=None):
        """Initialise the Locator. If you pass in a file path "example.wav" here it will call self.load(example.wav).

        Arguments:
            mic_locations (Mx2 tuple): Matrix of microphone coordinates, in meters, relative to the center of the array.
            file_path (None OR string): If present, will call self.load() on the given file path with the default load parameters
        """
        self.mics = np.array(mic_locations)
        self._mic_pairs_ = np.array(
            [p for p in multiset_combinations(np.arange(0, self.mics.shape[0], dtype="int16"), n=2)])
        self.maxdist = np.max(np.linalg.norm(self.mics, axis=1))
        self.spatial_nyquist_freq = self.sound_speed/(2*self.maxdist)

        if file_path:
            self.load(file_path)

    def load(self, file_path, normalise: bool=True, GCC_processor="CC", do_FFTs=True, filterpairs=False):
        """Loads the data from the .wav file, and computes the intra-channel correlations.

        Correlations are computed and then interpolated and stored within the locator
        object for use in the optimisation function or wherever necessary. Pass in a numpy array of data rather than
        loading from a file by setting raw_data = True.

        Arguments:
            file_path (str): The file path of the WAV file to be read.
            normalise (bool): Normalise the data? This is a good idea, hence the truthy default state.
            GCC_processor (str): Which GCC processor to use. Options are: CC, PHAT, Scot, & RIR. See Table 1 of Knapp, C. et. al. (1976) "The Generalised Correlation Method for Estimation of Time Delay"
            do_FFTs (bool): Calculate the cross-correlations? Worth turning off to save time if only MUSIC-based algorithms are to be used.
            filterpairs (bool): If true, filters out all spectral components above the spatial nyquist frequency for each pair of micrphones.
        """
        global wf, pftw
        self._GCC_proc_ = GCC_processor
        if isinstance(file_path, str):
            self.sample_rate, data = wav.read(file_path)
        else:
            data = file_path[1]
            self.sample_rate = file_path[0]
        # Convert from integer array to floating point to allow for computation
        data = data.astype('float64')

        # Normalise the data
        if normalise:
            for i in range(data.shape[1]):
                data[:, i] -= data[:, i].mean()


        # Store appropriately
        self.data = data
        # self._whiten_signal_()

        if do_FFTs:

            # if filterpairs:
            #     for pr in self._mic_pairs_:
            #         fc = self.sound_speed / (2 * np.linalg.norm(self.mics[pr[0],:], self.mics[pr[1],:]))
            #         w = fc / (self.sample_rate / 2)
            #         b, a = signal.butter(5, w, 'low')
            #         output = signal.filtfilt(b, a, self.data, axis=0)
            #         self.data = output
            #         return

            temp_pad = np.concatenate(
                (data, np.zeros(((2**(np.ceil(np.log2(data.shape[0])))-data.shape[0]).astype('int32'), data.shape[1]))),
                0)
            c = 1
            for prdx in np.arange(0, self._mic_pairs_.shape[0]):
                pr = self._mic_pairs_[prdx, :]
                self._cor_fns_["{}".format(pr)] = self._create_interp_(self.mics[pr[0], :], self.mics[pr[1], :],
                                                                     temp_pad[:, pr[0]], temp_pad[:, pr[1]],
                                                                          filteraliased=filterpairs)
                c += 1
            if pftw and not wf:
                with open('pyfftw_wisdom.txt', 'wb') as f:
                    pickle.dump(pyfftw.export_wisdom(), f)
                wf = True

    def filter_aliased(self):
        """Filters out all frequencies above the spatial Nyquist frequency for the current array confguration.
        """
        fc = self.spatial_nyquist_freq
        w = fc / (self.sample_rate / 2)
        if w >= 1:
            return
        b, a = signal.butter(5, w, 'low')
        output = signal.filtfilt(b, a, self.data, axis=0)
        self.data = output
        return

    def _whiten_signal_(self):

        for idx in np.arange(self.mics.shape[0]):
            t = np.fft.rfft(self.data[:,idx])
            self.data[:, idx] = np.fft.irfft(t/np.abs(t), n=2*len(t)-1)

    # @jit(nopython=True)
    def _create_interp_(self, mic1, mic2, mic1data, mic2data, buffer_percent=-10.0, res_scaling=5, filteraliased=False):
        """This function is to create the cubic interpolants for use in the correlation function. Uses GCC.

        Arguments:
            mic1 (int): The index of the first microphone of interest.
            mic2 (int): The index of the second microphone of interest.
            mic1data (np.array): The data corresponding to the first microphone.
            mic2data (np.array): The data corresponding to the second microphone.
            buffer_percent (float): The percent headroom to give the correlation function to avoid out-of-range exceptions.
            res_scaling (float): Scales the resolution
        """
        if filteraliased:
            fc = self.sound_speed / (2 * np.linalg.norm(mic1 - mic2))
            w = fc / (self.sample_rate / 2)
            if w <= 1:
                b, a = signal.butter(5, w, 'low')

                mic1data = signal.filtfilt(b, a, mic1data, axis=0)
                mic2data = signal.filtfilt(b, a, mic2data, axis=0)

        dlen = len(mic1data)
        num_samples = la.norm(mic1-mic2)*(1+buffer_percent/100.0)*(1/self.sound_speed)*self.sample_rate
        num_samples = int(round(num_samples))
        if buffer_percent < 0:
            num_samples = dlen-1

        n = 2*dlen
        X1 = fft_pack.rfft(mic1data, n=n)

        X2 = fft_pack.rfft(mic2data, n=n)
        X2star = np.conj(X2)

        # Implement more processors (Eckart, ML/HT)
        if self._GCC_proc_== "PHAT":
            corr = fft_pack.irfft(np.exp(1j*np.angle(X1 * X2star)), n=(res_scaling * n))

        elif self._GCC_proc_== "p-CSP" or self._GCC_proc_== "p-PHAT":
            proc = 1.0/(np.abs(X1*X2star)**0.73)
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        elif self._GCC_proc_== "CC":
            proc = 1.0
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        elif self._GCC_proc_== "RIR":
            proc = 1.0/(X1*np.conj(X1))
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        elif self._GCC_proc_== "SCOT":
            proc = 1.0/sqrt((X1*np.conj(X1))*(X2*X2star))
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        elif self._GCC_proc_== "HB":
            proc = np.abs(X1*np.conj(X2))/(X1*np.conj(X1)*X2*np.conj(X2))
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        elif (self._GCC_proc_ == "ML") or (self._GCC_proc_ == "HT"):
            # NOT WORKING, CHECKING SOMETHING
            raise Warning("This one isn't working. Don't use it.")
            gamma12 = (X1*X2star)/sqrt((X1*np.conj(X1))*(np.conj(X2star)*X2star))
            nmsqgm = gamma12*np.conj(gamma12)
            proc = nmsqgm/((X1*X2star)*(1-nmsqgm))
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        else:
            proc = 1.0
            corr = fft_pack.irfft(X1 * X2star * proc, n=(res_scaling * n))

        # corr = fft_pack.irfft(X1*X2star*proc, n=(res_scaling*n))

        corr = np.concatenate((corr[-int(res_scaling*n/2):], corr[:int(res_scaling*n/2)+1]))

        corrxs = np.arange(start=(dlen-num_samples)*res_scaling, stop=(dlen+num_samples)*res_scaling, step=1,
                           dtype='int32')

        cInterp = interp1d(x=(corrxs/res_scaling-dlen)+1, y=corr[corrxs], kind='cubic')

        return cInterp


    def _create_interp_NA_(self, mic1, mic2, mic1data, mic2data, buffer_percent=0.1, filteraliased=True):
        """This function is to create the cubic interpolants for use in the correlation function. Uses GCC WITHOUT the Fourier Transform to avoid aliasing.

        Arguments:
            mic1 (int): The index of the first microphone of interest.
            mic2 (int): The index of the second microphone of interest.
            mic1data (np.array): The data corresponding to the first microphone.
            mic2data (np.array): The data corresponding to the second microphone.
            buffer_percent (float): The percent headroom to give the correlation function to avoid out-of-range exceptions.
            res_scaling (float): Scales the resolution
        """

        dlen = len(mic1data)
        num_samples = la.norm(mic1-mic2)*(1+buffer_percent)*(1/self.sound_speed)*self.sample_rate
        num_samples = int(round(num_samples))
        if buffer_percent < 0:
            num_samples = dlen-1

        corr = signal.fftconvolve(mic1data, mic2data[::-1])
        nl = (len(corr)-1)//2
        corr = corr[nl-num_samples:nl+num_samples+1]

        corrxs = np.arange(start=-num_samples, stop=num_samples+1, step=1, dtype='int32')

        cInterp = interp1d(x=corrxs, y=corr, kind='cubic')

        return cInterp

    def _objective_(self, X, Y):
        """This function takes a matrix/vector of each x and y coordinates, and at each location evaluates the sum of the generalised
        cross-correlations between the microphone data as if the signal had come from that location. In this way we can search
        for the point with maximum correlation, which should correspond to the most likely actual source position.

        Args:
            X (np.array): An n by m matrix of x-coordinates of points at which to evaluate the _objective_ function
            Y (np.array): An n by m matrix of Y-coordinates of points at which to evaluate the _objective_ function

        Returns:
            np.array: An n by m matrix of signal correlations corresponding to the source having originated at each point
            generated by the input coordinates
        """

        # Calculate distances
        ds = [sqrt((X-self.mics[i, 0])**2+(Y-self.mics[i, 1])**2) for i in
              np.arange(0, self.mics.shape[0], dtype="int16")]

        # Calculate times, then pass the times into the correlation functions and sum them
        ts = np.array([self._cor_fns_["{}".format(ps)]((ds[ps[0]] - ds[ps[1]]) * self.sample_rate / self.sound_speed) for ps in
                       self._mic_pairs_])

        return np.sum(ts, axis=0)

    def display_heatmap(self, xrange=(-50, 50), yrange=(-50, 50), xstep=False, ystep=False, colormap="gist_heat", shw=True,
                        block_run=True, no_fig=False):
        """Displays a heatmap for visual inspection of correlation-based location estimation.

        Generates a grid of provided dimension/resolution, and evaluates the optimisation function at each point on the grid.
        Vectorised for fast execution.

        Arguments:
            xrange (float, float): The lower and upper bound in the x-direction.
            yrange (float, float): The lower and upper bound in the y-direction.
            xstep (float): If given, determines the size of the steps in the x-direction. Otherwise defaults to 1000 steps.
            ystep (float): If given, determines the size of the steps in the y-direction. Otherwise defaults to 1000 steps.
            colormap (str): The colour map for the heatmap. See https://matplotlib.org/examples/color/colormaps_reference.html
            shw (bool): If False, return the axis object rather than display.
            block_run (bool): Pause execution of the file while the figure is open? Set to True for running in the command-line.
            no_fig (bool): If True, return the heatmap grid rather than plot it.

        Returns:
            np.array: Returns EITHER the current (filled) heatmap domain if no_fig == True, OR a handle to the displayed figure.
        """

        if (xstep and ystep):
            xdom = np.linspace(start=xrange[0], stop=xrange[1], num=int((xrange[1] - xrange[0])//xstep))
            ydom = np.linspace(start=yrange[0], stop=yrange[1], num=int((yrange[1] - yrange[0])//ystep))
        else:
            xdom = np.linspace(start=xrange[0], stop=xrange[1], num=1000)
            ydom = np.linspace(start=yrange[0], stop=yrange[1], num=1000)
        self._hm_domain_ = np.zeros((len(ydom), len(xdom)))

        xdom, ydom = np.meshgrid(xdom, ydom)
        self._hm_domain_ = self._objective_(xdom, ydom)

        if no_fig:
            return self._hm_domain_

        plt.imshow(self._hm_domain_, cmap=colormap, interpolation='none', origin='lower',
                   extent=[xrange[0], xrange[1], yrange[0], yrange[1]])
        plt.colorbar()
        plt.xlabel("Horiz. Dist. from Center of Array [m]")
        plt.ylabel("Vert. Dist. from Center of Array [m]")
        plt.title("{} Processor".format(self._GCC_proc_))
        if shw:
            plt.show(block=block_run)
            return
        else:
            return plt.imshow(self._hm_domain_, cmap=colormap, interpolation='none', origin='lower',
                              extent=[xrange[0], xrange[1], yrange[0], yrange[1]])

    def display_radial(self, radius=10000, npoints=360, bearing=None, block_run=True, shw=True):
        """Display a polar plot of correlation as a function of angle.

        Arguments:
            radius (float): The radius at which to evaluate the _objective_ function. Defaults to 10km for far-field approximation.
            npoints (int): The total number of points around the circle at which to evaluate the _objective_ function.
            bearing (int): The magnetic bearing of microphone 1, if known. Will adjust axes to include NESW bearings. If left blank, will default to degrees relative to the arm housing microphone 1.
            block_run (bool): Pause execution of the file while the figure is open? Set to True for running in the command-line.
            shw (bool): If True, shows the figure, if False, returns the data.
        """

        points = np.zeros((npoints, 2))
        xs = 2*np.pi*np.arange(start=0, stop=npoints)/npoints
        points[:, 0] = radius*np.cos(xs)
        points[:, 1] = radius*np.sin(xs)

        cors = self._objective_(points[:, 0], points[:, 1])
        if shw:
            if bearing!=None:
                cors = np.roll(cors, round((-bearing/360)*npoints))
                ax = plt.subplot(111, projection='polar')
                ax.plot(xs, cors)
                ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
                ax.set_yticklabels([])
                ax.set_title("Estimated Acoustic Source Bearing at r={:.3f}m".format(radius))
            else:
                ax = plt.subplot(111, projection='polar')
                ax.plot(xs, cors)
                ax.set_xticklabels(['        0$^{\circ}$', '45$^{\circ}$', '90$^{\circ}$', '135$^{\circ}$',
                                    '180$^{\circ}$', '225$^{\circ}$', '270$^{\circ}$', '315$^{\circ}$'])
                ax.set_yticklabels([])
                ax.set_title("Estimated Source Direction, r={:.3f}m".format(radius))
            plt.show(block=block_run)
        else:
            return cors

    def polynom_steervec(self, samples, max_tau=1500):
        """Takes a vector of M desired delays and a maximum lag parameter max_tau, and returns the (M, 1, 2*max_tau+1)
        polynomial steering vector which contains the z-transform of the fractional delay filters necessary to delay each
        channel of a M-channel audio file by the corresponding delay parameter from the input vector. For example,
        polynom_steervec([1.2, 3, np.pi]) may be used to delay the first channel of a three-channel audio clip by 1.2 samples,
        the second channel by 3, and the third channel by approximately pi samples.

        Arguments:
            samples (np.array): A 1D array of the desired delay amounts
            max_tau (int): The maximum lag in either direction for the fractional delay filters. A higher number will be more accurate, but take longer to use.

        Returns:
            np.array: A vector containing the desired fractional delay filters.
        """
        mics = self.mics
        tau = samples
        tau.shape = (tau.shape[0], 1)
        Az = np.sinc((np.tile(np.arange(-max_tau, max_tau + 1), (mics.shape[0], 1)) - tau))
        Az = np.reshape(Az, (mics.shape[0], 1, Az.shape[-1]))
        return Az

    def shift_sound(self, location, inputfile, output_filename, noisescale=0):
        """Creates a multi-track wav file from a mono one, simulating the array recordings had the sound came from the
        provided location and were recorded on the current virtual array's microphone configuration.

        Saves the resultant file in the current working directory with the provided filename, at the same sample rate as the input data.

        Arguments:
            location (float, float): A tuple (x,y) providing the location in meters relative to the array at which to simulate the sound as having came from.
            inputfile (str): File path for the mono wav file to be used.
            output_filename (str): The desired file name for the output file.
            noisescale (float): Adds Gaussian white noise with standard deviation noisescale*(standard deviation of input file)
        """

        [spl, dt] = wav.read(inputfile)
        dt = (dt - np.mean(dt))/np.std(dt)
        loc_dif = self.mics - np.tile(location, (self.mics.shape[0], 1))
        dists = np.linalg.norm(loc_dif, axis=1)
        samples = (dists*spl)/self.sound_speed
        # print(self.mics, "\n", loc_dif, "\n", dists, samples, samples-min(samples), self.maxdist*spl/self.sound_speed, self.sound_speed, spl)
        samples -= min(samples)

        fracsamples = samples % 1
        # print(fracsamples)
        intsamples = samples - fracsamples
        # print(intsamples)

        svs = self.polynom_steervec(fracsamples)
        # print(svs.shape)
        # plt.plot(svs[:,0,1500-25:1500+25].T)
        # plt.show()

        t = np.tile(dt.T, (self.mics.shape[0], 1))
        t.shape = (t.shape[0], 1, t.shape[1])

        xout = []
        for r in np.arange(t.shape[0]):
            xout.append(fftconvolve(t[r, 0, :], svs[r, 0, :]))

        xout = np.array(xout)

        xout = np.hstack((xout, np.zeros((xout.shape[0], int(np.max(intsamples))))))

        for idx in np.arange(xout.shape[0]):
            xout[idx,:] = np.roll(xout[idx,:], int(intsamples[idx]))


        if noisescale != 0:
            xout += np.random.normal(0, noisescale*sqrt(np.var(xout)), size=xout.shape)

        xout *= (2**15-1)/np.max(abs(xout))

        xout = xout.astype('int16')
        wav.write(output_filename, spl, xout.T)

    def _MUSIC1D_(self, freqtup, theta, numsignals=1, SI=None):
        """Vectorised implementation of the Multiple Signal Classification algorithm for DOA eastimation.

        Arguments:
            freqtup (float, int): The frequency at which to evaluate the MUSIC algorithm, and the index at where to find it in the FFT of the data.
            theta (float/np.array): The angle of arrial at which to evaluate the MUSIC algorithm. May be a 1D numpy array.
            numsignals (int): How many signals to localise.
            SI (np.array): The covariance matrix S, if known a priori.
        """
        freq, idx = freqtup

        incidentdir = np.array([-np.cos(theta), -np.sin(theta)])
        tau = dot(self.mics, incidentdir) / self.sound_speed

        # Populate a(theta)
        a = np.exp(-1j * 2 * np.pi * freq * tau)

        # Find variance/covariance matrix S=conj(X.X^H)
        # Where X = FFT(recieved vector x)
        if type(SI) == type(None):
            S = dot(self.dataFFT[idx:idx+1, :].T, np.conj(self.dataFFT[idx:idx+1, :]))
        else:
            S = SI

        # Find eigen-stuff of S
        lam, v = la.eig(S)

        # Should be real as S Hermitian, rounding problems
        # mean imaginary part != 0. Take real part.
        lam = lam.real

        # Find a sorting index list
        xs = argsort(lam).astype("int16")
        # print(lam[xs])

        # Take the Eigenvectors corresponding to the
        # 'numsignals' lowest Eigenvalues
        EN = v[:, xs[:len(xs)-numsignals]]

        # Calculate 1/P_MU
        p = dot(dot(np.conj(a.T), EN), dot(np.conj(EN.T), a))

        # If more than 1D find relevant entries and flatten
        if len(p.shape) > 1:
            p = np.ndarray.flatten(np.diag(p))

        # Return P_MU
        return 1/p.real, lam[xs[-1]]

    def _MUSIC2D_(self, freqtup, X, Y, numsignals=1, SI=None):
        """Vectorised 2D implementation of the Multiple Signal Classification algorithm for DOA eastimation.

        Arguments:
            freqtup (float, int): The frequency at which to evaluate the MUSIC algorithm, and the index at where to find it in the FFT of the data.
            X (np.array): An array of x-locations at which to evaluate. Should be the counterpart to Y, as in np.meshgrid
            Y (np.array): An array of y-locations at which to evaluate. Should be the counterpart to X, as in np.meshgrid
            numsignals (int): How many signals to localise.
            SI (np.array): The covariance matrix S, if known a priori.
        """
        crds = np.dstack((X, Y))
        crds = np.stack([crds for _ in range(self.mics.shape[0])], 3)
        delm = np.linalg.norm(crds[:, :]-self.mics.T, axis=2)/self.sound_speed
        freq, idx = freqtup

        # Populate a(r, theta)
        # print(freq)
        # freq = freq*2*np.pi/self.sample_rate
        # print(freq)
        a = np.exp(-1j*2*np.pi*freq*delm)

        # Find variance/covariance matrix S=X.X^H
        # Where X = FFT(recieved vector x)
        if type(SI)==type(None):
            S = dot(self.dataFFT[idx:idx+1, :].T, np.conj(self.dataFFT[idx:idx+1, :]))
        else:
            S = SI

        # Find eigen-stuff of S
        lam, v = la.eigh(S)

        # Should be real as S Hermitian, rounding problems
        # mean imaginary part != 0. Take real part.
        lam = lam.real

        # Find a sorting index list
        xs = argsort(lam).astype("int16")

        # Take the Eigenvectors corresponding to the
        # 'numsignals' lowest Eigenvalues
        EN = v[:, xs[:len(xs)-numsignals]]

        # Calculate 1/P_MU
        p = dot(a, np.conj(EN))*dot(np.conj(a), EN)
        p = np.sum(p, axis=-1, keepdims=False)

        # Return P_MU
        return 1/p.real

    def display_radMUSIC(self, frequency, npoints=360, signals=1, shw=True, block_run=False):
        """Display a polar plot of estimated DOA using the MUSIC algorithm

        Arguments:
            frequency (float): The frequency to use for the MUSIC calculation.
            npoints (int): The total number of points around the circle at which to evaluate.
            signals (int): The numbers of signals to locate.
            shw (bool): Show the plot? If False, will return the data that was to be plotted.
            block_run (bool): Pause execution of the file while the figure is open? Set to True for running in the command-line.
        """

        X = 2*np.pi*np.arange(start=0, stop=npoints)/npoints
        self.dataFFT = fft_pack.rfft(self.data, axis=0, n=2*self.data.shape[0])
        pos = fft_pack.rfftfreq(2*self.data.shape[0])*self.sample_rate
        actidx = np.argmin(abs(pos-frequency))
        cors, _ = self._MUSIC1D_((pos[actidx], actidx), X, numsignals=signals)

        if shw:
            plt.figure()
            ax = plt.subplot(111, projection='polar')
            ax.plot(X, cors)
            ax.set_title("Estimated Acoustic Source Direction")
            plt.show(block=block_run)
        else:
            return cors

    def display_radMUSICchunked(self, frequency, npoints=360, signals=1, shw=True, block_run=False, chunks=10):
        """Display a polar plot of estimated DOA using the MUSIC algorithm

        Arguments:
            frequency (float): The frequency to use for the MUSIC calculation.
            npoints (int): The total number of points around the circle at which to evaluate.
            signals (int): The numbers of signals to locate.
            shw (bool): Show the plot? If False, will return the data that was to be plotted.
            block_run (bool): Pause execution of the file while the figure is open? Set to True for running in the command-line.
        """

        X = 2*np.pi*np.arange(start=0, stop=npoints)/npoints

        tRxx = np.zeros((self.mics.shape[0], self.mics.shape[0]), dtype='complex128')
        indices = [int(x) for x in np.linspace(0, self.data.shape[0], num=chunks + 1, endpoint=True)]
        for mark in np.arange(len(indices) - 1):
            dcr = self.data[indices[mark]:indices[mark + 1], :]
            print(dcr.shape)
            ft = fft_pack.rfft(dcr, axis=0, n=2*self.data.shape[0])
            pos = fft_pack.rfftfreq(2 * self.data.shape[0]) * self.sample_rate
            actidx = np.argmin(abs(pos - frequency))
            tRxx += np.dot(ft[actidx:actidx+1].T, np.conj(ft[actidx:actidx+1]))
        tRxx /= chunks
        print(tRxx)

        # self.dataFFT = fft_pack.rfft(self.data, axis=0, n=2*self.data.shape[0])

        cors, _ = self._MUSIC1D_((pos[actidx], actidx), X, numsignals=signals, SI=tRxx)

        if shw:
            plt.figure()
            ax = plt.subplot(111, projection='polar')
            ax.plot(X, cors)
            ax.set_title("Estimated Acoustic Source Direction")
            plt.show(block=block_run)
        else:
            return cors

    def display_MUSICheatmap(self, frequency, signals=1, xrange=(-50, 50), yrange=(-50,50), xstep=False, ystep=False,
                             colormap="gist_heat", shw=True, block_run=False, no_fig=False, title=""):
        """Displays a heatmap for visual inspection of MUSIC-based location estimation.

        Generates a grid of provided dimension/resolution, and evaluates the MUSIC algorithm at the given frequency at
        each point on the grid. Vectorised for fast execution.

        Arguments:
            frequency (float): The frequency (in Hz) at which to evaluate the MUSIC algorithm.
            xrange (float, float): The lower and upper bound in the x-direction.
            yrange (float, float): The lower and upper bound in the y-direction.
            xstep (float): If given, determines the size of the steps in the x-direction. Otherwise defaults to 1000 steps.
            ystep (float): If given, determines the size of the steps in the y-direction. Otherwise defaults to 1000 steps.
            colormap (str): The colour map for the heatmap. See https://matplotlib.org/examples/color/colormaps_reference.html
            shw (bool): If False, return the axis object rather than display.
            block_run (bool): Pause execution of the file while the figure is open? Set to True for running in the command-line.
            no_fig (bool): If True, return the heatmap grid rather than plot it.

        Returns:
            np.array: Returns EITHER the current (filled) heatmap domain if no_fig == True, OR a handle to the displayed figure.
        """
        if (xstep and ystep):
            xdom = np.linspace(xrange[0], xrange[1], num=(xrange[1] - xrange[0])//xstep)
            ydom = np.linspace(yrange[0], yrange[1], num=(yrange[1] - yrange[0])//ystep)
        else:
            xdom = np.linspace(start=xrange[0], stop=xrange[1], num=1000)
            ydom = np.linspace(start=yrange[0], stop=yrange[1], num=1000)

        self._hm_domain_ = np.zeros((len(ydom), len(xdom)))
        self.dataFFT = fft_pack.rfft(self.data, axis=0, n=2*self.data.shape[0])
        pos = fft_pack.rfftfreq(2*self.data.shape[0])*self.sample_rate
        actidx = np.argmin(abs(pos-frequency))
        ff = (pos[actidx], actidx)
        xdom, ydom = np.meshgrid(xdom, ydom)

        self._hm_domain_ = self._MUSIC2D_(ff, xdom, ydom, numsignals=signals)

        if no_fig:
            return self._hm_domain_

        f = plt.figure()
        plt.imshow(self._hm_domain_, cmap=colormap, interpolation='none', origin='lower',
                   extent=[xrange[0], xrange[1], yrange[0], yrange[1]])
        plt.colorbar()
        plt.xlabel("Horiz. Dist. from Center of Array [m]")
        plt.ylabel("Vert. Dist. from Center of Array [m]")

        if shw:
            plt.show(block=block_run)
            # f.savefig(title, bbox_inches='tight')
            # plt.close(f)
            return
        else:
            return f

    def display_MUSICheatmapchunked(self, frequency, signals=1, xrange=(-50, 50), yrange=(-50,50), xstep=False, ystep=False,
                             colormap="gist_heat", shw=True, block_run=False, no_fig=False, title="", chunks=10):
        """Displays a heatmap for visual inspection of MUSIC-based location estimation.

        Generates a grid of provided dimension/resolution, and evaluates the MUSIC algorithm at the given frequency at
        each point on the grid. Vectorised for fast execution.

        Arguments:
            frequency (float): The frequency (in Hz) at which to evaluate the MUSIC algorithm.
            xrange (float, float): The lower and upper bound in the x-direction.
            yrange (float, float): The lower and upper bound in the y-direction.
            xstep (float): If given, determines the size of the steps in the x-direction. Otherwise defaults to 1000 steps.
            ystep (float): If given, determines the size of the steps in the y-direction. Otherwise defaults to 1000 steps.
            colormap (str): The colour map for the heatmap. See https://matplotlib.org/examples/color/colormaps_reference.html
            shw (bool): If False, return the axis object rather than display.
            block_run (bool): Pause execution of the file while the figure is open? Set to True for running in the command-line.
            no_fig (bool): If True, return the heatmap grid rather than plot it.

        Returns:
            np.array: Returns EITHER the current (filled) heatmap domain if no_fig == True, OR a handle to the displayed figure.
        """
        if (xstep and ystep):
            xdom = np.linspace(xrange[0], xrange[1], num=(xrange[1] - xrange[0])//xstep)
            ydom = np.linspace(yrange[0], yrange[1], num=(yrange[1] - yrange[0])//ystep)
        else:
            xdom = np.linspace(start=xrange[0], stop=xrange[1], num=1000)
            ydom = np.linspace(start=yrange[0], stop=yrange[1], num=1000)

        self._hm_domain_ = np.zeros((len(ydom), len(xdom)))
        self.dataFFT = fft_pack.rfft(self.data, axis=0, n=2*self.data.shape[0])
        pos = fft_pack.rfftfreq(2*self.data.shape[0])*self.sample_rate
        actidx = np.argmin(abs(pos-frequency))
        ff = (pos[actidx], actidx)
        xdom, ydom = np.meshgrid(xdom, ydom)

        tRxx = np.zeros((self.mics.shape[0], self.mics.shape[0]), dtype='complex128')
        indices = [int(x) for x in np.linspace(0, self.data.shape[0], num=chunks + 1, endpoint=True)]
        for mark in np.arange(len(indices) - 1):
            dcr = self.data[indices[mark]:indices[mark + 1], :]
            print(dcr.shape)
            ft = fft_pack.rfft(dcr, axis=0, n=2 * self.data.shape[0])
            pos = fft_pack.rfftfreq(2 * self.data.shape[0]) * self.sample_rate
            actidx = np.argmin(abs(pos - frequency))
            tRxx += np.dot(ft[actidx:actidx + 1].T, np.conj(ft[actidx:actidx + 1]))
        tRxx /= chunks

        self._hm_domain_ = self._MUSIC2D_(ff, xdom, ydom, numsignals=signals, SI=tRxx)

        if no_fig:
            return self._hm_domain_

        f = plt.figure()
        plt.imshow(self._hm_domain_, cmap=colormap, interpolation='none', origin='lower',
                   extent=[xrange[0], xrange[1], yrange[0], yrange[1]])
        plt.colorbar()
        plt.xlabel("Horiz. Dist. from Center of Array [m]")
        plt.ylabel("Vert. Dist. from Center of Array [m]")

        if shw:
            plt.show(block=block_run)
            # f.savefig(title, bbox_inches='tight')
            # plt.close(f)
            return
        else:
            return f


    def _transpose_(self, mult):
        """
        """
        assert self.data is not None, "No data loaded yet."
        datatr = np.zeros((self.data.shape[0]*mult, self.data.shape[1]))
        tt = self.data.shape[0]/self.sample_rate
        fc = self.sample_rate/(2*mult)
        w = fc / (self.sample_rate / 2)
        b_bl, a_bl = signal.butter(10, w, 'low')
        for ch in np.arange(self.data.shape[1]):
            sig = interp1d(np.linspace(0, tt, num=self.data.shape[0], endpoint=True),
                           self.data[:,ch], kind="cubic", assume_sorted=True)
            d = sig(np.linspace(0, tt, self.sample_rate*mult*tt, endpoint=True))
            datatr[:, ch] = signal.filtfilt(b_bl, a_bl, d, axis=0)
            # datatr[:, ch] = d
        wav.write(data=datatr.astype('int16'), rate=self.sample_rate, filename="./scarynoise.wav")
        _, dnew = wav.read("./scarynoise.wav")
        return dnew

    def _UfitoRyy_(self, Rxx, f):
        """Returns the covariance matrix of the data at frequency f (Hz), shifted to the focussing frequency f_0. These
        should be summed over all frequencies of interest to create the universally focussed sample covariance matrix
        R_{yy} for the AF-MUSIC algorithm.

        Arguments:
            f (int): The frequency index to work with from the FFT of the data
        """
        df = self.dataFFT[:, f:f+1]
        ta = dot(df, df.conj().T)
        ui, Ufi = eigh(Rxx, check_finite=False)
        ui = ui.real
        sortarg = argsort(abs(ui))[::-1]
        Ufi = Ufi[:, sortarg]
        Tauto = (dot(self.Uf0, Ufi.conj().T)) / sqrt(self.numbins)
        Y = dot(Tauto, df)
        Ryy = dot(Y, Y.conj().T)
        return Ryy*abs(ui[sortarg[0]])


    def AF_MUSIC2D(self, focusing_freq=-1, signals=1, xrange=(-50, 50), yrange=(-50, 50), xstep=False, ystep=False,
                   colormap="gist_heat", shw=True, block_run=True, no_fig=False, chunks=10):
        """Displays a heatmap for visual inspection of AF-MUSIC-based location estimation.

        Generates a grid of provided dimension/resolution, and evaluates the AF-MUSIC algorithm at
        each point on the grid.

        Arguments:
            focusing_freq (float): The frequency (in Hz) at which to perform the calculation. If <0, will default to 0.9*(spatial Nyquist frequency)
            signals (int): The number of signals to locate.
            xrange (float, float): The lower and upper bound in the x-direction.
            yrange (float, float): The lower and upper bound in the y-direction.
            xstep (float): If given, determines the size of the steps in the x-direction. Otherwise defaults to 1000 steps.
            ystep (float): If given, determines the size of the steps in the y-direction. Otherwise defaults to 1000 steps.
            colormap (str): The colour map for the heatmap. See https://matplotlib.org/examples/color/colormaps_reference.html
            shw (bool): If False, return the axis object rather than display.
            block_run (bool): Pause execution of the file while the figure is open? Set to True for running in the command-line.
            no_fig (bool): If True, return the heatmap grid rather than plot it.

        Returns:
            np.array: Returns EITHER the current (filled) heatmap domain if no_fig == True, OR a handle to the displayed figure.
        """
        raise NotImplementedError

        self.dataFFT = fft_pack.rfft(self.data, axis=0).T
        self.numbins = self.dataFFT.shape[1]
        pos = fft_pack.rfftfreq(self.data.shape[0])*self.sample_rate

        if focusing_freq < 0:
            focusing_freq = self.spatial_nyquist_freq*0.45
            # focusing_freq = pos[np.argmax(self.dataFFT[0, :])]
        else:
            focusing_freq = 1000

        idxs = np.array(np.arange(pos.shape[0]))

        refidx = np.argmin(abs(pos-focusing_freq))
        Rcoh = np.zeros((self.dataFFT.shape[0], self.dataFFT.shape[0]), dtype='complex128')
        ul, self.Uf0 = la.eigh(dot(self.dataFFT[:, refidx:refidx+1], self.dataFFT[:, refidx:refidx+1].conj().T)/self.dataFFT.shape[0])
        ul = ul.real
        self.Uf0 = self.Uf0[:,argsort(abs(ul))[::-1]]


        pool = Pool(processes=7)
        res = pool.map(self._UfitoRyy_, idxs)
        pool.close()
        for r in res:
            Rcoh += r

        if xstep and ystep:
            xdom = np.linspace(start=xrange[0], stop=xrange[1], num=int((xrange[1] - xrange[0])//xstep))
            ydom = np.linspace(start=yrange[0], stop=yrange[1], num=int((yrange[1] - yrange[0])//ystep))
        else:
            xdom = np.linspace(start=xrange[0], stop=xrange[1], num=1000)
            ydom = np.linspace(start=yrange[0], stop=yrange[1], num=1000)
        self._hm_domain_ = np.zeros((len(ydom), len(xdom)))

        xdom, ydom = np.meshgrid(xdom, ydom)
        self._hm_domain_ = self._MUSIC2D_((pos[refidx], refidx), xdom, ydom, numsignals=signals, SI=Rcoh)

        if no_fig:
            return self._hm_domain_

        f = plt.figure()
        plt.imshow(self._hm_domain_, cmap=colormap, interpolation='none', origin='lower',
                   extent=[xrange[0], xrange[1], yrange[0], yrange[1]])
        plt.colorbar()
        plt.xlabel("Horiz. Dist. from Center of Array [m]")
        plt.ylabel("Vert. Dist. from Center of Array [m]")

        if shw:
            plt.show(block=block_run)
            return
        else:
            return f

    def AF_MUSIC(self, focusing_freq=-1, npoints=1000, signals=1, shw=True, block_run=True, chunks=10):
        """Display a polar plot of estimated DOA using the MUSIC algorithm

        Arguments:
            focusing_freq (float): The frequency (in Hz) at which to perform the calculation. If <0, will default to 0.9*(spatial Nyquist frequency)
            npoints (int): The total number of points around the circle at which to evaluate.
            signals (int): The numbers of signals to locate.
            shw (bool): Show the plot? If False, will return the data that was to be plotted.
            block_run (bool): Pause execution of the file while the figure is open? Set to True for running in the command-line.
            chunks (int): How many sections to split the data up into. Will split up the data and average the result over the split sections
        """
        omegas = np.linspace(0, 2 * np.pi, npoints, endpoint=False)
        rest = np.zeros_like(omegas)

        if focusing_freq < 0:
            focusing_freq = self.spatial_nyquist_freq*0.9

        # First generate Rxxs to get to T_autos
        # Tauto will go in here
        Tauto = np.zeros((self.mics.shape[0], self.mics.shape[0], self.data.shape[0]//2+1) , dtype="complex128")
        # Rxx will go in here
        Rxx = np.zeros((self.mics.shape[0], self.mics.shape[0], self.data.shape[0]//2+1) , dtype="complex128")
        # Ufi will go in here
        Ufi = np.zeros((self.mics.shape[0], self.mics.shape[0], self.data.shape[0]//2+1) , dtype="complex128")
        # Ryy will go in here
        Ryy = np.zeros((self.mics.shape[0], self.mics.shape[0], self.data.shape[0] // 2 + 1), dtype="complex128")

        # Split the data up into "chunks" sections
        indices = [int(x) for x in np.linspace(0, self.data.shape[0], num=chunks+1, endpoint=True)]

        # Calculate Rxx
        for mark in np.arange(len(indices)-1):
            dcr = self.data[indices[mark]:indices[mark+1], :]
            for chnl in np.arange(dcr.shape[1]):
                dcr[:, chnl] *= np.blackman(dcr.shape[0])

            # dft is RFFT of current data chunk
            dft = fft_pack.rfft(dcr, axis=0, n=self.data.shape[0]).T
            dft.shape = (dft.shape[0], 1, dft.shape[1])
            # print(dft.shape, self.data.shape, Tauto.shape)

            Rxx += np.einsum("jin,iln->jln", dft, np.conj(np.transpose(dft, (1,0,2))))/chunks

        # The frequencies for Tauto and DFT. They all have the same length so this is fine to do outside the loop
        pos = fft_pack.rfftfreq(self.data.shape[0]) * self.sample_rate

        # focusing_freq_index is the index along dft and Tauto to find f_0
        focusing_freq_index = np.argmin(np.abs(pos - focusing_freq))

        eig_f0, v_f0 = np.linalg.eigh(Rxx[:,:,focusing_freq_index])
        Uf0 = v_f0[:, np.argsort(np.abs(eig_f0))[::-1]]

        # Calculate Tautos
        for indx, fi in enumerate(pos):
            eig_fi, v_fi = np.linalg.eigh(Rxx[:, :, indx])
            Ufi[:,:,indx] = v_fi[:, np.argsort(np.abs(eig_fi))[::-1]]
            Tauto[:,:,indx] = dot(Uf0, np.conj(Ufi[:,:,indx].T))/np.sqrt(pos.shape[0])

        # Calculate Ryy
        chunks=1.0
        indices = [int(x) for x in np.linspace(0, self.data.shape[0], num=chunks + 1, endpoint=True)]
        for mark in np.arange(len(indices) - 1):
            dcr = self.data[indices[mark]:indices[mark + 1], :]
            for chnl in np.arange(dcr.shape[1]):
                dcr[:, chnl] *= np.blackman(dcr.shape[0])

            # dft is RFFT of current data chunk
            dft = fft_pack.rfft(dcr, axis=0, n=self.data.shape[0]).T
            dft.shape = (dft.shape[0], 1, dft.shape[1])
            # print(dft.shape, self.data.shape, Tauto.shape)

            Yi = np.einsum("abc,bdc->adc", Tauto, dft)
            Ryy += np.einsum("jin,iln->jln", Yi, np.conj(np.transpose(Yi, (1, 0, 2)))) / chunks

        Rcoh = np.sum(Ryy, axis=-1)/(self.data.shape[0]//2+1)

        rest, _ = self._MUSIC1D_((focusing_freq, focusing_freq_index), omegas, SI=Rcoh)

        if shw:
            plt.figure()
            ax = plt.subplot(111, projection='polar')
            ax.plot(omegas, rest)
            ax.set_title("Estimated Acoustic Source Direction")
            plt.show(block=block_run)
        else:
            return rest

    def _broadband_MUSIC_(self, max_tau, resolution=720, delta=0.0001, max_iter=1000):
        """This was where I was implementing the broadband MUSIC algorithm."""

        raise Exception("This is untested and unfinished. SBR2 was not able to diagonalise the STCM well enough to continue.")

        pstcm = self._space_time_covm_(max_tau)
        Q, gs, r2s = SBR2(pstcm, delta=delta, maxiter=max_iter, loss=1.0e-6)
        import pickle
        with open('Hsave', 'wb') as f:
            pickle.dump(Q, f)
        with open('gsave', 'wb') as f:
            pickle.dump(gs, f)
        with open('r2save', 'wb') as f:
            pickle.dump(r2s, f)
        D = pmm(Q, pmm(pstcm, Q.h()))
        Qn = Q[:, 1:, :]
        try:
            qm = Q.max_tau
        except:
            print("Had to force Q.max_tau")
            Q.max_tau = max_tau
        circlen = resolution
        musicspec = np.zeros((circlen, 2))
        idx = 0
        for theta in np.linspace(start=0, stop=2 * np.pi, num=circlen):
            atheta = self.polynom_steervec(theta=theta, max_tau=Q.max_tau)
            ah_Hn = pmm(atheta.h(), Qn)
            Hnh_a = pmm(Qn.h(), atheta)
            Gamma = pmm(ah_Hn, Hnh_a).flatten()
            musicspec[idx, 0] = theta
            musicspec[idx, 1] = 1.0 / Gamma[Q.max_tau].real
            idx += 1
        return musicspec, gs, r2s

    def _space_time_covm_(self, max_tau):
        """Generates the Space-Time Covariance Matrix (STCM), with maximum lag parameter max_tau.
        Adapted from PEVDToolbox/SpaceTimeCovMat.m from PEVD toolbox by S Weiss.

        Arguments:
            max_tau (int): The maximum lag parameter for the STCM.
        """
        M = self.data.shape[1]
        L = self.data.shape[0]
        R = PolynoMat(np.zeros((M, M, 2*max_tau+1), dtype='complex128'))
        Lm = L-2*max_tau

        Xref = self.data[max_tau:max_tau+Lm, :]
        for tau in np.arange(0, 2*max_tau+1):
            Xshift = self.data[tau:tau+Lm, :]
            R[:, :, 2*max_tau-tau] = dot(Xref.T, Xshift)/Lm
        R = (R+R.H())/2
        return R

    def estimate_DOA(self, npoints=1000):
        """Gives an estimate of the source DOA based on AF-MUSIC for frequencies below the spatial Nyquist frequency, and
        a non aliasing version of GCC for frequencies above the spatial Nyquist frequency.

        Arguments:
            npoints (int): The number of points around the circle to evaluate the DOA estimation.
        """
        fc = self.spatial_nyquist_freq
        w = fc / (self.sample_rate / 2)
        b_bl, a_bl = signal.butter(10, w, 'low')

        data_below_SNF = np.zeros_like(self.data)
        data_above_SNF = np.zeros_like(self.data)

        for idx in np.arange(self.data.shape[1]):
            data_above_SNF[:, idx] = self.data[:, 0]
            data_below_SNF[:, idx] = signal.filtfilt(b_bl, a_bl, self.data[:, 0], axis=0)

        out_bl = self.AF_MUSIC(shw=False, npoints=npoints)
        out_abv = self.display_radial(100000, shw=False, npoints=npoints)

        ax = plt.subplot(111, projection='polar')
        ax.plot(np.linspace(0, 2*np.pi, npoints), (out_bl / np.max(np.abs(out_bl)))*(out_abv / np.max(np.abs(out_abv))))
        plt.show()
        return


def UCA(n, r, centerpoint=True, show=False):
    """A helper function to easily set up UCAs (uniform circular arrays).

    Arguments:
        n (int): The number of microphones in the array.
        r (float): The radius of the array
        centerpoint (bool): Include a microphone at (0,0)? This will be one of the n points.
        show (bool): If True, shows a scatterplot of the array

    Returns:
        np.array: An n by 2 numpy array containing the x and y positions of the n microphones in the UCA.
    """
    mics = []
    if centerpoint:
        n -= 1
        mics.append([0,0])
    for theta in np.linspace(0, 2*np.pi, n, endpoint=False):
        mics.append([r*np.cos(theta), r*np.sin(theta)])
    mics = np.array(mics)
    if show:
        plt.scatter(mics[:,0], mics[:,1])
        plt.title("Microphone Locations")
        plt.xlabel("Horizontal Distance From Array Center [m]")
        plt.ylabel("Vertical Distance From Array Center [m]")
        plt.show()
    return mics
