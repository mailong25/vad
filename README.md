# Voice Activity Detection

## Credit claiming
The code is taken from this great repo https://github.com/hcmlab/vadnet

## Dependencies
- tensorflow==1.15.0
- gunicorn==19.4.5

sudo apt-get install gunicorn3

## Load model
```
import numpy as np
import requests
from vad import VAD
detector = VAD(frame_duration = 0.5, model_path = 'models/vad')
FRAME_SAMPLING_RATE = 44100
```
Where:
- frame_duration: how to split the frames/audio. For example, if frame_duration=0.5 and len(audio) = 5s, then the audio will be splited into 10 small chunks. Each chunk is 0.5s long. The model will run predictions on each chunk to check if it is a speech or silent. Make sure 0.1 < frame_duration < 1.0. The higher the frame_duration the better the accuracy
- model_path: path to model
- FRAME_SAMPLING_RATE: the sampling rate of the audio/frames.

## Make prediction from audio path
```
result = detector.predict('test.wav',FRAME_SAMPLING_RATE)
```

## Make prediction from raw bytes
```
with open('test.wav','rb') as f:
    header = f.read(44)
    frames = f.read()

array_frames = np.frombuffer(frames,dtype=np.int16)
array_frames = array_frames.astype(np.float32, order='C') / 32768.0
result = detector.predict(array_frames,FRAME_SAMPLING_RATE)
```
Note: The "array_frames" must be the type of numpy float array\
Here how you can convert frames in other types into numpy float array:

##### numpy int array -->  numpy float array
```
array_frames = int_frames.astype(np.float32, order='C') / 32768.0
```
#### bytes -->  numpy float array
```
array_frames = np.frombuffer(byte_frames,dtype=np.int16)
array_frames = array_frames.astype(np.float32, order='C') / 32768.0
```

# Host a inference server

Edit server.py, change the frame_duration and FRAME_SAMPLING_RATE.\
Make sure that the client only send binary data with the same sampling as FRAME_SAMPLING_RATE in server.py

Run the server
```
gunicorn3 -w 2 -t 3000 --threads 2 -b 0.0.0.0:5700 server:app
```

# Client call
```
with open('test.wav','rb') as f:
    header = f.read(44)
    frames = f.read()
```
Note: Make sure that the sampling rate of test.wav as the same as FRAME_SAMPLING_RATE in server.py

```
res = requests.post('http://127.0.0.1:5700/predict', data = frames)
res = res.json()['result']
```
