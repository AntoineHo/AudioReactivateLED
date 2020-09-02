import numpy as np
from numpy import abs, append, arange, insert, linspace, log10, round, zeros

class ExpFilter:
	"""Simple exponential smoothing filter"""
	def __init__(self, val=0.0, alpha_decay=0.5, alpha_rise=0.5):
		"""Small rise / decay factors = more smoothing"""
		assert 0.0 < alpha_decay < 1.0, 'Invalid decay smoothing factor'
		assert 0.0 < alpha_rise < 1.0, 'Invalid rise smoothing factor'
		self.alpha_decay = alpha_decay
		self.alpha_rise = alpha_rise
		self.value = val

	def update(self, value):
		if isinstance(self.value, (list, np.ndarray, tuple)):
			alpha = value - self.value
			alpha[alpha > 0.0] = self.alpha_rise
			alpha[alpha <= 0.0] = self.alpha_decay
		else:
			alpha = self.alpha_rise if value > self.value else self.alpha_decay
		self.value = alpha * value + (1.0 - alpha) * self.value
		return self.value
		
	def __str__(self,) :
		return """ExpFilter(alpha_decay={}, alpha_rise={})""".format(self.alpha_decay, self.alpha_rise)

"""
def frames_per_second():
    # Return the estimated frames per second
    # Returns the current estimate for frames-per-second (FPS).
    # FPS is estimated by measured the amount of time that has elapsed since
    # this function was previously called. The FPS estimate is low-pass filtered
    # to reduce noise.
    # This function is intended to be called one time for every iteration of
    # the program's main loop.
    # Returns
    # -------
    # fps : float
    #     Estimated frames-per-second. This value is low-pass filtered
    #     to reduce noise.
    
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)
"""

def sinewave(x, A=1, B=1, C=0, D=0) : # period is 2*pi*B
	return A * np.sin(B * (x + C) ) + D

def memoize(function):
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper

@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)

def interpolate(y, new_length):
    """Intelligently resizes the array by linearly interpolating the values
    Parameters
    ----------
    y : np.array
        Array that should be resized
    new_length : int
        The length of the new interpolated array
    Returns
    -------
    z : np.array
        New array with length of new_length that contains the interpolated
        values of y.
    """
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z


def rfft(data, window=None):
	window = 1.0 if window is None else window(len(data))
	ys = np.abs(np.fft.rfft(data * window))
	xs = np.fft.rfftfreq(len(data), 1.0 / config.MIC_RATE)
	return xs, ys


def fft(data, window=None):
	window = 1.0 if window is None else window(len(data))
	ys = np.fft.fft(data * window)
	xs = np.fft.fftfreq(len(data), 1.0 / config.MIC_RATE)
	return xs, ys

def hertz_to_mel(freq):
	"""Returns mel-frequency from linear frequency input.
	Parameter
	---------
	freq : scalar or ndarray
	Frequency value or array in Hz.
	Returns
	-------
	mel : scalar or ndarray
	Mel-frequency value or ndarray in Mel
	"""
	return 2595.0 * log10(1 + (freq / 700.0))


def mel_to_hertz(mel):
	"""Returns frequency from mel-frequency input.
	Parameter
	---------
	mel : scalar or ndarray
		Mel-frequency value or ndarray in Mel
	Returns
	-------
	freq : scalar or ndarray
	Frequency value or array in Hz.
	"""
	return 700.0 * (10**(mel / 2595.0)) - 700.0


def melfrequencies_mel_filterbank(num_bands, freq_min, freq_max, num_fft_bands):
	"""Returns centerfrequencies and band edges for a mel filter bank
	Parameters
	----------
	num_bands : int
		Number of mel bands.
	freq_min : scalar
		Minimum frequency for the first band.
	freq_max : scalar
		Maximum frequency for the last band.
	num_fft_bands : int
		Number of fft bands.
	Returns
	-------
	center_frequencies_mel : ndarray
	lower_edges_mel : ndarray
	upper_edges_mel : ndarray
	"""

	mel_max = hertz_to_mel(freq_max)
	mel_min = hertz_to_mel(freq_min)
	delta_mel = abs(mel_max - mel_min) / (num_bands + 1.0)
	frequencies_mel = mel_min + delta_mel * arange(0, num_bands + 2)
	lower_edges_mel = frequencies_mel[:-2]
	upper_edges_mel = frequencies_mel[2:]
	center_frequencies_mel = frequencies_mel[1:-1]
	return center_frequencies_mel, lower_edges_mel, upper_edges_mel


def compute_melmat(num_mel_bands=24, freq_min=64, freq_max=8000,
                   num_fft_bands=513, sample_rate=16000):
	"""Returns tranformation matrix for mel spectrum.
	Parameters
	----------
	num_mel_bands : int
		Number of mel bands. Number of rows in melmat.
		Default: 24
	freq_min : scalar
		Minimum frequency for the first band.
		Default: 64
	freq_max : scalar
		Maximum frequency for the last band.
		Default: 8000
	num_fft_bands : int
		Number of fft-frequenc bands. This ist NFFT/2+1 !
		number of columns in melmat.
		Default: 513   (this means NFFT=1024)
	sample_rate : scalar
		Sample rate for the signals that will be used.
		Default: 44100
	Returns
	-------
	melmat : ndarray
		Transformation matrix for the mel spectrum.
		Use this with fft spectra of num_fft_bands_bands length
		and multiply the spectrum with the melmat
		this will tranform your fft-spectrum
		to a mel-spectrum.
	frequencies : tuple (ndarray <num_mel_bands>, ndarray <num_fft_bands>)
		Center frequencies of the mel bands, center frequencies of fft spectrum.
	"""

	center_frequencies_mel, lower_edges_mel, upper_edges_mel =  \
		melfrequencies_mel_filterbank(num_mel_bands, freq_min, freq_max, num_fft_bands)

	center_frequencies_hz = mel_to_hertz(center_frequencies_mel)
	lower_edges_hz = mel_to_hertz(lower_edges_mel)
	upper_edges_hz = mel_to_hertz(upper_edges_mel)
	freqs = linspace(0.0, sample_rate / 2.0, num_fft_bands)
	melmat = zeros((num_mel_bands, num_fft_bands))

	for imelband, (center, lower, upper) in enumerate(zip(center_frequencies_hz, lower_edges_hz, upper_edges_hz)):

		left_slope = (freqs >= lower) == (freqs <= center)
		melmat[imelband, left_slope] = ( (freqs[left_slope] - lower) / (center - lower) )

		right_slope = (freqs >= center) == (freqs <= upper)
		melmat[imelband, right_slope] = ( (upper - freqs[right_slope]) / (upper - center) )

	return melmat, (center_frequencies_mel, freqs)
