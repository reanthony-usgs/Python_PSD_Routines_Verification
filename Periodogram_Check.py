#!/usr/bin/env python

# This code is a Sanity check of various periodogram algorithms
# (without any zero padding) including:

# 1) From numpy's FFT routine
# 2) SciPy's Periodogram routine 
# 3) Mlab's CSD routine

# All codes agree to machine precision 

# Verified using different lengths of nfft and different sample rate data

# Note that using mlab CSD, NFFT must be greater than length of Timeseries
#(points)/2 to get a periodogram out - otherwise it will attempt section 
# averaging 

# Using different detrending schemes on the different methodologies appears 
# to have a very minimal impact on the calculated PSDs in these Test Cases 



from obspy.core import read, UTCDateTime
from scipy.signal import periodogram, welch, csd
from matplotlib.mlab import csd as csd2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def rms(A):
    return np.sqrt(np.mean(np.abs(A)**2))


debug = False


# font parameters for plots
font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)


net, sta, loc, chan = "IU", "ANMO", "00", "LHZ"

# Number of points of windows and overlap for PSD calculations 
nfft = 1000
windlap = 0.
# This is defining my boxcar window for mlab CSD
My_Window = np.ones(nfft);
# Set the start time 
stime = UTCDateTime("2015-206T00:00:00")

########### Start of Actual Code 


# Set the path to the data
path = "/msd/" + net + "_" + sta + "/" + str(stime.year) + "/" + str(stime.julday).zfill(3) 
path += "/" + loc + "_" + chan + "*.seed"
st = read(path)

# Get length of record (T) in seconds, sampling rate (fs) in Hz, and number of data points (N)
T = nfft*st[0].stats.delta
fs = 1./st[0].stats.delta 

# Let's detrend the full day like seedscan
st.detrend(type='linear')

# First use np.rfft code to get Periodogram
st_FFT = st.copy()

####### Option - Triming Timeseries ####################################
# same number of points as nfft
# etime = stime + T - (1*st[0].stats.delta) 

#6 hour window 
etime = UTCDateTime("2015-206T06:00:00")

########## End Option ################################################## 

st_FFT.trim(stime,etime)

# Now let's detrend like SciPy Welch defaults to 
st_FFT.detrend('constant')

# Let's get just the data
tr_FFT = st_FFT[0]

if debug:
    print(fs)

# Steps are

#1 do FFT,
#2 Convert to Power  
#3 Normalize. Since we are only considering the positive frequencies, double 

# Note the fft routine does not take in a frequency. Therefore the square of the fourier transform has units of energy. 
# We must normalize NOT by 1/T as is standard convention for power from the FFT (e.g., Aster and Borchers, 2013) but by 1/(N*fs)
# Where fs is the sample rate and N is the total number of samples.

FT = np.fft.rfft(tr_FFT.data, n=nfft)


if debug:
    print('Here is the Length of the data:', len(tr_FFT.data))
    print('Here is nfft:', nfft)


# Sanity check to make sure we get the same RMS in both time and frequency domains

# Note - as nfft becomes smaller, this sanity check becomes more invalid as noise begins to dominate (Adam, email exchange)
#This doesn't work for small lengths because of the following issues: 
#Parseval's theorem only holds for square integrable functions defines as a sum of sines and cosines. 
#When you only have a few points, then you are only getting out a few Fourier coefficients and you 
#are dominated by noise, with noise being the deviation from a pure square integrable function with sines and cosines.


TS_RMS = rms(tr_FFT.data)
FT_RMS = rms(FT)
FT_RMS = (1./np.sqrt(nfft))*rms(FT)

RMS_Diff = TS_RMS-FT_RMS

print('Difference in RMS between timeseries and FT is:', RMS_Diff)

FT_P = 2.*(1./(nfft*fs))*(FT**2.)

# get the Frequency Vector 
FT_F = np.fft.rfftfreq(nfft, d=tr_FFT.stats.delta)

# Get rid of DC term
FT_P = FT_P[1:]
FT_F = FT_F[1:]


########## Now we Repeat using (more) canned Python packages


##### Process data the same way

# trim to the selected data
st.trim(stime,etime)
# Let's get just the data
tr = st[0]

##### Option - Which canned PSD program? ###############################

#Toggle These to go between SciPy Periogram and mlab CSD 

# Scipy Periodogram Code 
#freq, power = periodogram(tr.data, fs = fs, nfft=nfft)   

#mplotlib CSD
power,freq = csd2(tr.data, tr.data, NFFT=nfft, detrend='mean', Fs = 1./tr.stats.delta,
                    window=My_Window, noverlap=None)
                    
####### End Option #####################################################

freq, power = freq [1:], power[1:] 

# Convert all to dB
poweR_FT =10.*np.log10(np.abs(FT_P))
poweR = 10.*np.log10(np.abs(power))

fig = plt.figure(1)
plt.semilogx(1./freq, poweR, '.', label='Periodogram')
plt.semilogx(1./FT_F, poweR_FT, '.', label='My Code', linewidth=2)
plt.xlabel('Period (s)')
plt.ylabel('Power (dB rel. 1 count^2/Hz)')
plt.legend()

# Do the power difference and make a plot (let it default y-axis value)

# Compare on a linear scale 
Periodogram_Difference = 10**(poweR/10.) - 10**(poweR_FT/10.)

fig = plt.figure(2)
plt.semilogx(1./freq, Periodogram_Difference) 
plt.xlabel('Period (s)')
plt.ylabel('Power Difference (counts^2/Hz)') 
plt.show()
