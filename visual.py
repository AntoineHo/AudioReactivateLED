# SIGNAL PROCESSING
import numpy as np
from numpy import abs, append, arange, insert, linspace, log10, round, zeros
from scipy.ndimage.filters import gaussian_filter1d


# plot & save to file
import matplotlib.pyplot as plt
#global f, fr, fg, fb, fw, i
#f = open("fa.txt", "w")
#fr = open("fr.txt", "w")
#fg = open("fg.txt", "w")
#fb = open("fb.txt", "w")
#fw = open("fw.txt", "w")
i = 0


# Filters
from filters import *

import sys # DEBUG
import time

""" VISUALIZATION FUNCTIONS """
def rightshiftmean(ar, drift) :
    shift = int(drift / 2)
    if shift != 0 :
        out = np.empty_like(ar)
        out[:shift] = ar[-shift:]
        out[shift:] = ar[:-shift]
    else :
        out = ar
    ret = np.cumsum(out, dtype=float)
    ret[2:] = ret[2:] - ret[:-2]
    fin = np.append(ret[2 - 1:] / 2, ret[-1:])
    return fin

def visualize_scroll(y, o):
    """Effect that originates in the center and scrolls outwards"""
    y = y**2.0
    o.gain.update(y)
    y /= o.gain.value
    y *= 255.0
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))
    # Scrolling effect window
    o.p[:, 1:] = o.p[:, :-1]
    o.p = o.p * 0.98
    o.p = gaussian_filter1d(o.p, sigma=0.2)
    # Create new color originating at the center
    o.p[0, 0] = r
    o.p[1, 0] = g
    o.p[2, 0] = b
    # Update the LED strip
    return np.concatenate((o.p[:, ::-1], o.p), axis=1)

def visualize_energy(y, o):
    """Effect that expands from the center with increasing sound energy"""
    y = np.copy(y)
    o.gain.update(y)
    y /= o.gain.value
    # Scale by the width of the LED strip
    y *= float((o.NUM_PIXELS // 2) - 1)
    
    # Map color channels according to energy in the different freq bands
    scale = 0.998
    r = int(np.mean(y[:len(y) // 3]**scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = int(np.mean(y[2 * len(y) // 3:]**scale))
    
    # Assign color to different frequency regions
    o.p[0, :r] = 255.0
    o.p[0, r:] = 0.0
    o.p[1, :g] = 255.0
    o.p[1, g:] = 0.0
    o.p[2, :b] = 255.0
    o.p[2, b:] = 0.0
    o.p_filt.update(o.p)
    o.p = np.round(o.p_filt.value)
    # Apply substantial blur to smooth the edges
    o.p[0, :] = gaussian_filter1d(o.p[0, :], sigma=4.0)
    o.p[1, :] = gaussian_filter1d(o.p[1, :], sigma=4.0)
    o.p[2, :] = gaussian_filter1d(o.p[2, :], sigma=4.0)
    # Set the new pixel value
    return np.concatenate((o.p[:, ::-1], o.p), axis=1)

def visualize_spectrum(y, o):
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""
    y = np.copy(interpolate(y, o.NUM_PIXELS // 2))
    
    o.common_mode.update(y)
    diff = y - o._prev_spectrum
    o._prev_spectrum = np.copy(y)
    
    # Color channel mappings
    r = o.r_filt_strong.update(y - o.common_mode.value)
    g = np.abs(o.g_filt_strong.update(diff))
    #g = o.g_filt_strong.update(np.abs(diff))
    b = o.b_filt_strong.update(np.copy(y))
    w = o.w_filt_strong.update(np.copy(y))
    
    #print(w)
    
    # Mirror the color channels for symmetric output
    r = np.concatenate((r[::-1], r))
    g = np.concatenate((g[::-1], g))
    b = np.concatenate((b[::-1], b))
    w = np.concatenate((w[::-1], w))
    
    
    
    
    output = np.array([r, g, b, w]) * 255
    
    return output

def visualize_beat(y, o):
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""
    y = np.copy(y)
    o.gain.update(y)
    y /= o.gain.value
    # Scale by the width of the LED strip
    y *= float((o.NUM_PIXELS // 2) - 1)
    
    # Map color channels according to energy in the different freq bands
    scale = 0.94
    w = int(np.mean(y[:len(y) // 4]**scale)) # BASS
    g = int(np.mean(y[len(y) // 4: 2 * len(y) // 4]**scale))
    b = int(np.mean(y[2 * len(y) // 4: 3 * len(y) // 4]**scale))
    r = int(np.mean(y[3 * len(y) // 4:]**scale)) # HIGH
    
    # Assign color to different frequency regions
    o.z[0, :r] = 255.0
    o.z[0, r:] = 0.0
    o.z[1, :g] = 255.0
    o.z[1, g:] = 0.0
    o.z[2, :b] = 255.0
    o.z[2, b:] = 0.0
    o.z_filt.update(o.z)
    o.z = np.round(o.z_filt.value)
    
    # Apply substantial blur to smooth the edges
    o.z[0, :] = gaussian_filter1d(o.z[0, :], sigma=3.0)
    o.z[1, :] = gaussian_filter1d(o.z[1, :], sigma=3.0)
    o.z[2, :] = gaussian_filter1d(o.z[2, :], sigma=3.0)
    
    # Set the new pixel value
    return np.concatenate((o.z[:, ::-1], o.z), axis=1)
    
    output = np.array([g, r, b, w]) * 255
    return output

def visualize_play(y, o):
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""
    y = np.copy(interpolate(y, o.NUM_PIXELS // 2))
    
    o.common_mode.update(y)
    diff = y - o._prev_spectrum
    o._prev_spectrum = np.copy(y)
    
    # Color channel mappings
    if np.random.random() <= o.SWAP_FREQUENCY*2 :
        print('OOops I swapped')
        np.random.shuffle(o.color_order)
    
    r = o.r_filt.update(y - o.common_mode.value)
    g = np.abs(diff)
    g = o.g_filt.update(g)
    b = o.b_filt.update(y - o.common_mode.value)
    w = o.w_filt.update(y - o.w_mode.value)
    
    if np.random.random() <= o.SWAP_FREQUENCY :
        o.SWAP = not o.SWAP
        print("Swapped!")
        
    if not o.SWAP :
        # red
        r[:10] *= 1.5 #  <-- Very low
        #r[:10] = np.where(r[:10] > 1.0, r[:10], 1.0)
        # white
        w[10:30] *= 1.5 #  <-- BASS
        #w[10:30] = np.where(w[10:30] > 1.0, w[10:30], 1.0)
        w[:10] *= 0.15 # Quarter intensity of white in last 3/4 <-- Very low BASS
        w[30:] *= 0.15 # Quarter intensity of white in last 3/4 <-- BASS
        # green
        g[:30] *= 0.25 # Quarter intensity of green in first 1/4
        g[30:60] *= 1.5
        #g[30:60] = np.where(g[30:60] > 1.0, g[30:60], 1.0)
        g[60:] *= 0.25 # Quarter intensity of green in last half
        # blue
        b[:60] *= 0.25 # Quarter intensity of blue in first 3/4
        b[60:90] *= 1.5
        #b[60:90] = np.where(b[60:90] > 1.0, b[60:90], 1.0)
        b[90:] *= 0.25 # Quarter intensity of blue in first 3/4
        #red
        r[:90] *= 0.25 # Quarter intensity of white in first 3/4 <-- High
        r[90:] *= 1.5
        #r[90:] = np.where(r[90:] > 1.0, r[90:], 1.0)
    else :
        # white
        w[:10] *= 1.5 #  <-- Very low
        #w[:10] = np.where(w[:10] > 1.0, w[:10], 1.0)
        # red
        r[10:30] *= 1.5 #  <-- BASS
        #r[10:30] = np.where(r[10:30] > 1.0, r[10:30], 1.0)
        r[:10] *= 0.25 # Quarter intensity of white in last 3/4 <-- Very low BASS
        r[30:] *= 0.25 # Quarter intensity of white in last 3/4 <-- BASS
        # blue
        b[:30] *= 0.25 # Quarter intensity of green in first 1/4
        b[30:60] *= 1.5
        #b[30:60] = np.where(b[30:60] > 1.0, b[30:60], 1.0)
        b[60:] *= 0.25 # Quarter intensity of green in last half
        # green
        g[:60] *= 0.25 # Quarter intensity of blue in first 3/4
        g[60:90] *= 1.5
        #g[60:90] = np.where(g[60:90] > 1.0, g[60:90], 1.0)
        g[90:] *= 0.25 # Quarter intensity of blue in first 3/4
        # white
        w[:90] *= 0.15 # Quarter intensity of white in first 3/4 <-- High
        w[90:] *= 1.5
        #w[90:] = np.where(w[90:] > 1.0, w[90:], 1.0)

    # Smooth values in between ranges
    b = gaussian_filter1d(b, sigma=4.0)
    g = gaussian_filter1d(g, sigma=4.0)
    r = gaussian_filter1d(r, sigma=4.0)
    w = gaussian_filter1d(w, sigma=4.0)
    
    """
    # add travelers
    if o.traveler <= o.NUM_PIXELS // 2 :
        o.traveler += o.trav_speed
    else :
        o.traveler = 0
    
    start = o.traveler - o.trav_len
    start = 0 if start < 0 else start
    end = o.traveler
    end = o.NUM_PIXELS // 2 if end > o.NUM_PIXELS // 2 else end
    print(start, end)
    w[start:end] *= 10.0
    np.where(w[start:end] > 1.0, w[start:end], 1.0)
    """
    
    # Mirror the color channels for symmetric output + apply drift
    if o.DRIFT_RATE != 0 :
        o.DRIFTED += o.DRIFT_RATE
        
        if o.DRIFTED >= o.NUM_PIXELS * 2 :
            o.DRIFTED = 1

        r = rightshiftmean(np.concatenate((r[::-1], r)), o.DRIFTED)
        g = rightshiftmean(np.concatenate((g[::-1], g)), o.DRIFTED)
        b = rightshiftmean(np.concatenate((b[::-1], b)), o.DRIFTED)
        w = rightshiftmean(np.concatenate((w[::-1], w)), o.DRIFTED)
    else :
        r = np.concatenate((r[::-1], r))
        g = np.concatenate((g[::-1], g))
        b = np.concatenate((b[::-1], b))
        w = np.concatenate((w[::-1], w))
        
    # Apply sine wave to red :
    if o.frame_sine < len(o.y_sine) - 1 :
        o.frame_sine += 1
    else :
        o.frame_sine = 0
    
    if o.sine_drift < o.NUM_PIXELS :
        o.sine_drift += 1
    else :
        o.sine_drift = 0
        
    # Get current function
    current_frame = sinewave(o.x_sine, A=o.y_sine[o.frame_sine])
    current_frame = np.roll(current_frame, o.sine_drift)
    
    # Interpolate current_frame with the max volume in blue
    current_frame = np.interp(current_frame, (-1,1), (-1, np.max(b)))
    
    # Add to the red channel
    r = r + current_frame

    for col in [r,g,b,w] :
        col = np.where(col > 1.0, col, 1.0)
        col = np.where(col < 0.0, col, 0.0)
    
    dc = {0:g,1:r,2:b,3:w}
    colmap = [dc[i] for i in o.color_order]
    output = np.array(colmap) * 255

    return output
