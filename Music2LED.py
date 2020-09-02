# RASPBERRY AND LEDs
import board
import neopixel

# AUDIO SIGNAL HANDLING
import pyaudio
import wave
import audioop

# SIGNAL PROCESSING
import numpy as np
from numpy import abs, append, arange, insert, linspace, log10, round, zeros
from scipy.ndimage.filters import gaussian_filter1d

# SYSTEM / OS / SOFTWARE
import sys
import time
import argparse

### MODULES
from filters import *
from general import *
from visual import *

class Opt :
	def __init__(self, ) :
		parser = argparse.ArgumentParser(description="LED strip is responsive to sound!")
		parser.add_argument("-v", "--visual", type=str, default="scroll", choices=["energy", "scroll", "spectrum", "play", "beat"], help="Visualisation type. Default: %(default)s")
		parser.add_argument("-n", "--npixels", type=int, default=240, help="Number of LEDs on strip. Default: %(default)s")
		parser.add_argument("-b", "--brightness", type=self._restricted_float, default=0.8, metavar="(0.0, 1.0)", help="Brightness of LED strip. Default: %(default)s")
		parser.add_argument("-r", "--rate", type=int, default=44100, choices=[8000, 16000, 32000, 37800, 44100, 48000, 88200, 96000, 176400, 192000, 352800, 2822400], help="Input sample rate. Default: %(default)s")
		parser.add_argument("-F", "--FPS", type=int, default=30, help="Frames per second. Default: %(default)s")
		parser.add_argument("-c", "--channels", type=str, default="mono", choices=["mono", "stereo"], help="Mono or stereo. Default: %(default)s")
		parser.add_argument("-vol", "--volume", type=float, default=1e-5, help="Volume threshold of representation. Default: %(default)s")
		parser.add_argument("-rh", "--rolling_history", type=int, default=2, help="Rolling history of the strip. Default: %(default)s")
		parser.add_argument("-fft", "--fft", type=int, default=30, help="Fourier Fast Transform bin number. Default: %(default)s")
		parser.add_argument("-minf", "--min_frequency", type=int, default=200, help="Min frequency to create mel bank. Default: %(default)s")
		parser.add_argument("-maxf", "--max_frequency", type=int, default=8000, help="Max frequency to create mel bank. Default: %(default)s")
		parser.add_argument("-ad", "--alpha_decay", type=self._restricted_float, default=0.1, metavar="(0.0, 1.0)", help="Alpha decay used for the low-pass filters. Default: %(default)s")
		parser.add_argument("-ar", "--alpha_rise", type=self._restricted_float, default=0.9, metavar="(0.0, 1.0)", help="Alpha rise used for the low-pass filters. Default: %(default)s")
		parser.add_argument("-dr", "--drift_rate", type=int, default=0, help="Drift rate used in play visual style. Default: %(default)s")
		parser.add_argument("-np", "--num_peaks", type=int, default=6, help="Number of peaks in red sinewave. Default: %(default)s")
		
		args = parser.parse_args()
		
		# PARAMETRABLE OPTIONS
		self.VISUAL = args.visual
		self.NUM_PIXELS = args.npixels
		self.BRIGHTNESS = args.brightness
		self.SAMPLE_RATE = args.rate
		self.FPS = args.FPS
		self.CHANNELS = 1 if args.channels == "mono" else 2
		self.N_ROLLING_HISTORY = args.rolling_history
		self.MIN_VOLUME_THRESHOLD = args.volume
		self.N_FFT_BINS = args.fft
		self.ALPHA_DECAY = args.alpha_decay
		self.ALPHA_RISE = args.alpha_rise
		self.MIN_FREQUENCY = args.min_frequency
		self.MAX_FREQUENCY = args.max_frequency
		self.DRIFT_RATE = args.drift_rate
		self.NUM_PEAK_RED = args.num_peaks
		
		if self.VISUAL == "scroll" :
			self.visualization_effect = visualize_scroll
		elif self.VISUAL == "energy" :
			self.visualization_effect = visualize_energy
		elif self.VISUAL == "spectrum" :
			self.visualization_effect = visualize_spectrum
		elif self.VISUAL == "play" :
			self.visualization_effect = visualize_play
		elif self.VISUAL == "beat" :
			self.visualization_effect = visualize_beat
		else :
			self.visualization_effect = visualize_scroll
		
		print("""Parameters set:\n- Pixel number: {}\n- Style: {}\n- Brightness: {}\n- Sample rate: {}\n- Channels: {}\n- Rolling history: {}\n- FPS: {}\n- Volume threshold: {}\n- FFT bin number: {}\n- Default alpha decay: {}\n- Default alpha rise: {}\n- Min frequency: {}\n- Max frequency: {}""".format(self.NUM_PIXELS, self.VISUAL, self.BRIGHTNESS, self.SAMPLE_RATE, self.CHANNELS, self.N_ROLLING_HISTORY, self.FPS, self.MIN_VOLUME_THRESHOLD, self.N_FFT_BINS, self.ALPHA_DECAY, self.ALPHA_RISE, self.MIN_FREQUENCY, self.MAX_FREQUENCY))
		
		# NON PARAMETRABLE OPTIONS
		self._prev_spectrum = np.tile(0.01, self.NUM_PIXELS // 2)
		self._time_prev = time.time() * 1000.0 #  The previous time that the frames_per_second() function was called
		self.ORDER = neopixel.RGBW
		self.PIXEL_PIN = board.D18
		self.FORMAT = pyaudio.paInt16
		self.SWAP_FREQUENCY = 0.001
		self.SWAP = False
		self.DRIFTED = 0
		
		# SMOOTHING FILTERS
		self.fft_plot_filter = ExpFilter(np.tile(1e-1, self.N_FFT_BINS), alpha_decay=0.5, alpha_rise=0.99)
		print("Initialized FFT plot filter: {}".format(self.fft_plot_filter))
		self.mel_gain = ExpFilter(np.tile(1e-1, self.N_FFT_BINS), alpha_decay=0.01, alpha_rise=0.99)
		print("Initialized Mel gain filter: {}".format(self.mel_gain))
		print(self.mel_gain.value.shape)
		self.mel_smoothing = ExpFilter(np.tile(1e-1, self.N_FFT_BINS), alpha_decay=0.5, alpha_rise=0.99)
		print("Initialized Mel smoothing filter: {}".format(self.mel_smoothing))
		self.volume = ExpFilter(self.MIN_VOLUME_THRESHOLD, alpha_decay=0.02, alpha_rise=0.02)
		print("Initialized Volume filter: {}".format(self.volume))
		
		self.fft_window = np.hamming(int(self.SAMPLE_RATE / self.FPS) * self.N_ROLLING_HISTORY)
		self.prev_fps_update = time.time()
		
		self.r_filt = ExpFilter(np.tile(0.01, self.NUM_PIXELS // 2), alpha_decay=0.45, alpha_rise=0.85)
		print("Initialized Red filter: {}".format(self.r_filt))
		self.g_filt = ExpFilter(np.tile(0.01, self.NUM_PIXELS // 2), alpha_decay=0.55, alpha_rise=0.6)
		print("Initialized Green filter: {}".format(self.g_filt))
		self.b_filt = ExpFilter(np.tile(0.01, self.NUM_PIXELS // 2), alpha_decay=0.55, alpha_rise=0.6)
		print("Initialized Blue filter: {}".format(self.b_filt))
		self.w_filt = ExpFilter(np.tile(0.01, self.NUM_PIXELS // 2), alpha_decay=0.95, alpha_rise=0.95)
		print("Initialized White filter: {}".format(self.w_filt))
		self.w_mode = ExpFilter(np.tile(0.01, self.NUM_PIXELS // 2), alpha_decay=0.99, alpha_rise=0.01)

		self.common_mode = ExpFilter(np.tile(0.01, self.NUM_PIXELS // 2), alpha_decay=0.99, alpha_rise=0.01)
		self.common_mode_2 = ExpFilter(np.tile(0.01, self.NUM_PIXELS // 2), alpha_decay=0.95, alpha_rise=0.05)
		print("Initialized Common mode filter: {}".format(self.common_mode))
		self.p_filt = ExpFilter(np.tile(1, (4, self.NUM_PIXELS // 2)), alpha_decay=0.1, alpha_rise=0.99)
		print("Initialized p filter: {}".format(self.p_filt))
		self.p = np.tile(1.0, (4, self.NUM_PIXELS // 2))
		
		self.z_filt = ExpFilter(np.tile(1, (4, self.NUM_PIXELS // 2)), alpha_decay=0.5, alpha_rise=0.95)
		print("Initialized z filter: {}".format(self.p_filt))
		self.z = np.tile(1.0, (4, self.NUM_PIXELS // 2))
		
		self.gain = ExpFilter(np.tile(0.01, self.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
		print("Initialized Gain filter: {}".format(self.gain))
		print("Initialized all filters: Done")
		
		
		# Travelers
		self.traveler = 0
		self.trav_len = 30
		self.trav_speed = 3
		self.last_sent = 0
		
		# Red sinewave
		self.sine_drift = 0
		self.frame_sine = 0
		self.x_sine = np.linspace(0, self.NUM_PEAK_RED*np.pi, self.NUM_PIXELS)
		self.y_sine = np.arange(-1,1,0.05)
		self.y_sine = np.append(self.y_sine, self.y_sine[::-1])
		
		# Color order
		self.color_order = [0,1,2,3]
		
	def _restricted_float(self, f) :
		try :
			f = float(f)
		except ValueError :
			raise argparse.ArgumentTypeError("{} not a floating-point literal".format(f))
		if f < 0.0 or f > 1.0 :
			raise argparse.ArgumentTypeError("{} not in the acceptable range (0.0, 1.0)".format(f))
		return f

class MainLoop :
	def __init__(self, o) :
		
		# PIXELS
		self.strip = neopixel.NeoPixel(o.PIXEL_PIN, o.NUM_PIXELS, brightness=o.BRIGHTNESS, auto_write=False, pixel_order=o.ORDER)
		self._prev_pixels = np.tile(253, (4, o.NUM_PIXELS)) #Pixel values that were most recently displayed on the LED strip
		self.pixels = np.tile(1, (4, o.NUM_PIXELS)) # Pixel values for the LED strip	

		# AUDIO
		self.frames_per_buffer = int(o.SAMPLE_RATE / o.FPS)
		self.y_roll = np.random.rand(o.N_ROLLING_HISTORY, self.frames_per_buffer) / 1e16
		self.fft_window = o.fft_window # np.hamming(int(o.SAMPLE_RATE / o.FPS) * o.N_ROLLING_HISTORY)
		self.samples = None
		self.mel_y = None
		self.mel_x = None
		self.create_mel_bank(o)

		self.pa = pyaudio.PyAudio() # initialize PyAudio object
		self.stream_in = self.pa.open(format=o.FORMAT,
			channels=o.CHANNELS,
			rate=o.SAMPLE_RATE,
			input=True,
			frames_per_buffer=self.frames_per_buffer) # open stream object as input & output
		
		killer = GracefulKiller(self.pa, self.stream_in) # Initialize killer
		
		self.overflows = 0
		self.prev_ovf_time = time.time()

		while True :
			try:
				# y: audio_samples
				y = np.fromstring(self.stream_in.read(self.frames_per_buffer, exception_on_overflow=False), dtype=np.int16)
				y = y.astype(np.float32)
				self.stream_in.read(self.stream_in.get_read_available(), exception_on_overflow=False)
				self.update(y, o)
			except IOError:
				self.overflows += 1
				if time.time() > self.prev_ovf_time + 1:
					self.prev_ovf_time = time.time()
					print('Audio buffer has overflowed {} times'.format(self.overflows))
		
		self.stream_in.stop_stream()
		self.stream_in.close()
		self.pa.terminate()

	def update(self, audio_samples, o) :
		#global pixels, y_roll, mel_y, mel_x, fft_window, prev_rms, prev_exp, prev_fps_update, MIN_VOLUME_THRESHOLD
		
		# Normalize samples between 0 and 1
		y = audio_samples / 2.0**15
				
		# Construct a rolling window of audio samples
		self.y_roll[:-1] = self.y_roll[1:]
		self.y_roll[-1, :] = np.copy(y)
		y_data = np.concatenate(self.y_roll, axis=0).astype(np.float32)
		
		vol = np.max(np.abs(y_data))
		if vol < o.MIN_VOLUME_THRESHOLD : 
			print('No audio input. Volume below threshold. Volume:', vol)
			self.pixels = np.tile(0, (4, o.NUM_PIXELS))
			self.update_strip(o)
		
		else :
			# Transform audio input into the frequency domain
			N = len(y_data)
			N_zeros = 2**int(np.ceil(np.log2(N))) - N
			
			# Pad with zeros until the next power of two
			y_data *= o.fft_window
			y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
			YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
			
			# Construct a Mel filterbank from the FFT data
			mel = np.atleast_2d(YS).T * self.mel_y.T
			
			# Scale data to values more suitable for visualization
			mel = np.sum(mel, axis=0)
			mel = mel**2.0
			
			# Gain normalization
			o.mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
			mel /= o.mel_gain.value
			mel = o.mel_smoothing.update(mel)
			
			#print(mel)
			
			# Map filterbank output onto LED strip
			output = o.visualization_effect(mel, o)
			self.pixels = output
			self.update_strip(o)
			
	def update_strip(self, o) :
		p = np.clip(self.pixels, 0, 255).astype(int)
		out = np.column_stack((p[0].ravel(), p[1].ravel(), p[2].ravel(), p[3].ravel())).astype(int)
		for i in range(o.NUM_PIXELS):
			# Ignore pixels if they haven't changed (saves bandwidth)
			if np.array_equal(p[:, i], self._prev_pixels[:, i]) :
				continue
			self.strip[i] = out[i]

		self._prev_pixels = np.copy(p)
		self.strip.show()

	def create_mel_bank(self, o) :
		self.samples = int(o.SAMPLE_RATE * o.N_ROLLING_HISTORY / (2.0 * o.FPS))
		self.mel_y, (_, self.mel_x) = compute_melmat(num_mel_bands=o.N_FFT_BINS, freq_min=o.MIN_FREQUENCY, freq_max=o.MAX_FREQUENCY, num_fft_bands=self.samples, sample_rate=o.SAMPLE_RATE)
		
if __name__ == "__main__" :
	
	options = Opt()
	
	loop = MainLoop(options)
	
	sys.exit(0)
	
	
