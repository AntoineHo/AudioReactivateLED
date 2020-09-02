# AudioReactiveLED

Most of the code to process the audio signal was copied over from here (many thanks): https://github.com/scottlawsonbc/audio-reactive-led-strip I only adapted it to my setup.

```
usage: Music2LED.py [-h] [-v {energy,scroll,spectrum,play,beat}] [-n NPIXELS]
                    [-b 0.0, 1.0)]
                    [-r {8000,16000,32000,37800,44100,48000,88200,96000,176400,192000,352800,2822400}]
                    [-F FPS] [-c {mono,stereo}] [-vol VOLUME]
                    [-rh ROLLING_HISTORY] [-fft FFT] [-minf MIN_FREQUENCY]
                    [-maxf MAX_FREQUENCY] [-ad (0.0, 1.0)] [-ar (0.0, 1.0]
                    [-dr DRIFT_RATE] [-np NUM_PEAKS]

LED strip is responsive to sound!

optional arguments:
  -h, --help            show this help message and exit
  -v {energy,scroll,spectrum,play,beat}, --visual {energy,scroll,spectrum,play,beat}
                        Visualisation type. Default: scroll
  -n NPIXELS, --npixels NPIXELS
                        Number of LEDs on strip. Default: 240
  -b (0.0, 1.0), --brightness (0.0, 1.0)
                        Brightness of LED strip. Default: 0.8
  -r {8000,16000,32000,37800,44100,48000,88200,96000,176400,192000,352800,2822400}, --rate {8000,16000,32000,37800,44100,48000,88200,96000,176400,192000,352800,2822400}
                        Input sample rate. Default: 44100
  -F FPS, --FPS FPS     Frames per second. Default: 30
  -c {mono,stereo}, --channels {mono,stereo}
                        Mono or stereo. Default: mono
  -vol VOLUME, --volume VOLUME
                        Volume threshold of representation. Default: 1e-05
  -rh ROLLING_HISTORY, --rolling_history ROLLING_HISTORY
                        Rolling history of the strip. Default: 2
  -fft FFT, --fft FFT   Fourier Fast Transform bin number. Default: 30
  -minf MIN_FREQUENCY, --min_frequency MIN_FREQUENCY
                        Min frequency to create mel bank. Default: 200
  -maxf MAX_FREQUENCY, --max_frequency MAX_FREQUENCY
                        Max frequency to create mel bank. Default: 8000
  -ad (0.0, 1.0), --alpha_decay (0.0, 1.0)
                        Alpha decay used for the low-pass filters. Default:
                        0.1
  -ar (0.0, 1.0), --alpha_rise (0.0, 1.0)
                        Alpha rise used for the low-pass filters. Default: 0.9
  -dr DRIFT_RATE, --drift_rate DRIFT_RATE
                        Drift rate used in play visual style. Default: 0
  -np NUM_PEAKS, --num_peaks NUM_PEAKS
                        Number of peaks in red sinewave. Default: 6
```
