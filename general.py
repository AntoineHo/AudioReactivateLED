from signal import signal, SIGINT
import sys

class GracefulKiller :
	def __init__(self, pa, stream) :
		self.stream = stream
		self.pa = pa		
		signal(SIGINT, self.handler) # Signal handler for interruption
		
	def handler(self, signal_received, frame):
		# Handle any cleanup here
		print('\nExiting gracefully...')
		self.stream.stop_stream()
		self.stream.close()
		self.pa.terminate() # terminate pyaudio object
		sys.exit(0)
