import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import wave
import os

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hamming):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    


def logscale_spec(spec, sr=8000, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs


def moving_average(l, N):
	sum = 0
	result = list( 0 for x in l)
 
	for i in range( 0, N ):
		sum = sum + l[i]
		result[i] = sum / (i+1)
 
	for i in range( N, len(l) ):
		sum = sum - l[i-N] + l[i]
		result[i] = sum / N
 
	return result


def write_wave(path, audio, sample_rate, num_channels=1):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
    print('{} done!'.format(os.path.basename(path)))
    
    
def pcm2wav(pcm_file):
	with open(pcm_file,'rb') as f:
		str_data  = f.read()
	with wave.open(pcm_file[:-4]+'.wav','wb') as wave_out:
		wave_out.setnchannels(1)
		wave_out.setsampwidth(2)
		wave_out.setframerate(16000)
		wave_out.writeframes(str_data)
	print('format conversion done!')


def padding(segment,num_channels,sample_rate):
	pad = bytes([0 for i in range(sample_rate)])
	pad = bytearray(pad)
	segment_pad = 2*pad + bytearray(segment) + 2*pad
	segment_pad = bytes(segment_pad)
	return segment_pad


def plotstft(audiopath, binsize=2**15, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    print('sample length: ',len(samples))
    s = stft(samples, binsize)
    frame_length = binsize

    sshow, freq = logscale_spec(s, factor=20.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    ims = np.where(ims>180,ims,0)	# for those unvoiced frames with low frequency density, set them 0 for a more clear boundary.
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 10))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

    return s,ims,samplerate, frame_length, samples


def main():
	# pcm and wav file only
	filename = 'test05mi.pcm'
	if os.path.basename(filename).split(".")[1] == 'pcm':
		pcm2wav(filename)
		filename = filename[:-4] + '.wav'
	s,ims,samplerate, frame_length, data = plotstft(filename)
	mean_array = []
	for time in range(ims.shape[0]):
		freq_mean = np.mean(ims[time,:])
		mean_array.append(freq_mean)

	mean_array = moving_average(mean_array,5)

	# Attention: the threshold is adjustable, which should be judged manually by 
	# spectrogram(figure1) and frequency centroid graph(figure2)
	
	thres = 33
	endpoint = []
	for amp in range(len(mean_array)-1):
		if (mean_array[amp] - thres) * (mean_array[amp+1] - thres) <=0:
			endpoint.append(amp)
    
	plt.plot(mean_array)
	plt.show()
    
	print(endpoint,len(endpoint))

    # split * & record endpoint & write into files
	assert len(endpoint)%2 == 0
    # if not, you shall adjust the thres properly. Odd endpoints give broken intervals

	if not os.path.exists('./chunk'):
		os.mkdir('./chunk')
	for i in range(int(len(endpoint)/2)):
		chunk = data[int(endpoint[2*i]*frame_length/2):int(endpoint[2*i+1]*frame_length/2)]	# split
		chunk = padding(chunk,1,samplerate)	# padding 0
		write_wave('./chunk/chunk-{}.wav'.format(i),chunk,samplerate,1)	# write


if __name__ == '__main__':
	main()








    