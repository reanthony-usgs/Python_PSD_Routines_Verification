#!/usr/bin/env python


# This code is a sanity check showing that the mlab CSD code 
# (as implimented in the 2019 PSD techniques paper submitted to SRL) 
# does not systematically bias PSD estimates compared to the SciPy PSD 
# Algorithm (which was independently verified to be accurare at ASL by Dave
# Wilson and Rob Anthony (see Rob_Periodogram_Check.py) by comparison with 
# the numpy FFT routine. 

from obspy.core import read, UTCDateTime
from scipy.signal import welch, csd, windows
from matplotlib.mlab import csd as csd2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

debug = True


# font parameters for plots
font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)

net, sta, loc, chan = "IU", "ANMO", "00", "BHZ"

# Number of points of windows and overlap for PSD calculations 
nfft = 10000
windlap = 0.5

# This is defining my Hann window for mlab CSD
My_Window = windows.hann(nfft);

# Set the start time and end times - 6 hour window 
stime = UTCDateTime("2015-206T00:00:00")
etime = UTCDateTime("2015-206T06:00:00")

########### Start of Actual Code 


# Set the path to the data
path = "/msd/" + net + "_" + sta + "/" + str(stime.year) + "/" + str(stime.julday).zfill(3) 
path += "/" + loc + "_" + chan + "*.seed"
st = read(path)

# Get lsampling rate (fs) in Hz
fs = 1./st[0].stats.delta 

# Let's detrend the full day like seedscan
st.detrend(type='linear')

################## First use Scipy Welch #############################3
st_Welch = st.copy()
st_Welch.trim(stime,etime)
tr_Welch = st_Welch[0]

if debug:
    print(fs)


# Default window is a Hann ('Incorrectly called Hanning Taper")
# Default Detrend is de-mean

freq_Welch, power_Welch = welch(tr_Welch.data, fs = fs, nperseg = nfft,
                                noverlap = windlap*nfft)
                                

freq_Welch, power_Welch = freq_Welch[1:], power_Welch[1:] 



##### mlab CSD PSD #####################################################

# trim to the selected data
st.trim(stime,etime)

# Let's get just the data
tr = st[0]

power,freq = csd2(tr.data, tr.data, NFFT=nfft, Fs = 1./tr.stats.delta, 
                detrend='mean', window=My_Window,noverlap=windlap*nfft)


freq, power = freq [1:], power[1:] 

# Convert all to dB
poweR_Welch =10.*np.log10(np.abs(power_Welch))
poweR = 10.*np.log10(np.abs(power))

fig = plt.figure(1)
plt.semilogx(1./freq_Welch, poweR_Welch, '.', label='SciPy Welch')
plt.semilogx(1./freq, poweR, '.', label='Mlab CSD', linewidth=2)
plt.xlabel('Period (s)')
plt.ylabel('Power (dB rel. 1 count^2/Hz)')
plt.legend()
plt.show()


