# Voice Activity Detection

## Make inference
```
from vad import VAD
detector = VAD(frame_duration = 0.5, model_path = 'models/vad')
FRAME_SAMPLING_RATE = 44100

## Make prediction from raw bytes
with open('test.wav','wb') as f:
    header = f.read(44)
    frames = f.read()
```
If frames are represented as bytes --> Convert frames into numpy int array
```
array_frames = np.frombuffer(frames,dtype=np.int16)
result = detector.predict(array_frames,FRAME_SAMPLING_RATE)

## Make prediction from audio path
result = detector.predict(test.wav,FRAME_SAMPLING_RATE)

```

## Host in inference server

Edit server.py, change the frame_duration and FRAME_SAMPLING_RATE.\
Make sure that the client only send binary data with the same as FRAME_SAMPLING_RATE in server.py\

Run the server
```
gunicorn3 -w 2 -t 3000 --threads 2 -b 0.0.0.0:5700 server:app
```

# Client call
```
with open('test.wav','wb') as f:
    header = f.read(44)
    frames = f.read()
```

Note: Make sure that the sampling rate of test.wav as the same as FRAME_SAMPLING_RATE in server.py

```
res = requests.post('http://127.0.0.1:5700/predict', data = frames)
res = res.json()['result']
```

### 
array_frames = np.frombuffer(byte_frames,dtype=np.int16)

### If frames are represented as numpy float arry --> Convert frames into numpy int array
array_frames = (float_frames * 32768.0).astype(np.int16,order='C').tobytes()
