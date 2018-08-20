# Voice_Activity_Detection_V2
Voice Activity Detection Version2.0

It is because the testing file "test05mi.wav" has quite low SNR and energy that I attempt to cut the file in frequency domain. First thing we do is the Short Time Fourier Transform, which reflects all frequencies in a short frame along time axis. After summarizing the average and moving average, a threshold is manually choosen. Finally, 10 seperate files are saved!

# Usage
`Python vad_v2.py`

# Tips
You need to make sure the `.wav` file is mono-channel. Change the file name as well as all neccessary parameters in py script.

Spectrogram and its moving average figures are shown firstly. Check carefully if the boundary and threshold are properly set. (Or there will be odd endpoints which gives broken chunk parts!!)

1sec zeros are padded at the both ends of the frame. 

# Example
![image](https://github.com/guozhonghao1994/Voice_Activity_Detection_V2/blob/master/Figure_1.png)
![image](https://github.com/guozhonghao1994/Voice_Activity_Detection_V2/blob/master/Figure_1-1.png)

# Known Bugs
1. Some `.pcm` files may cause severe error.
I highly recommend the `.wav` file as input first.

2. When padding 0, in some rare cases, some harsh noise instead of silence are padded.

