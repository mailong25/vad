import sys, os, json, argparse, glob
import tensorflow as tf
import numpy as np
import librosa as lr
from tqdm import tqdm
import ntpath
import librosa
import numpy as np
import soundfile as sf
import time
import sys
import numpy as np

def audio_from_file(path, sr=None, ext=''):
    return lr.load('{}{}'.format(path, ext), sr=sr, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type='kaiser_best')                

def audio_to_file(path, x, sr):    
    lr.output.write_wav(path, x.reshape(-1), sr, norm=False)

def audio_to_frames(x, n_frame, n_step=None):    

    if n_step is None:
        n_step = n_frame

    if len(x.shape) == 1:
        x.shape = (-1,1)

    n_overlap = n_frame - n_step
    n_frames = (x.shape[0] - n_overlap) // n_step       
    n_keep = n_frames * n_step + n_overlap

    strides = list(x.strides)
    strides[0] = strides[1] * n_step

    return np.lib.stride_tricks.as_strided(x[0:n_keep,:], (n_frames,n_frame), strides)

class VAD:
    def __init__(self, frame_duration = 0.5, model_path = 'models/vad'):
        
        if frame_duration > 1.01:
            print('frame duration must lesser than 1(s)')
            sys.exit(0)
        
        path = model_path
        n_batch=256
        self.sampling_rate = 48000
        self.frame_size = int(round(self.sampling_rate * frame_duration))
        self.frame_duration = frame_duration
        
        print('load model from {}'.format(path))

        if os.path.isdir(path):
            candidates = glob.glob(os.path.join(path, 'model.ckpt-*.meta'))
            if candidates:
                candidates.sort()                
                checkpoint_path, _ = os.path.splitext(candidates[-1])
        else:
            checkpoint_path = path

        if not all([os.path.exists(checkpoint_path + x) for x in ['.data-00000-of-00001', '.index', '.meta']]):
            print('ERROR: could not load model')
            raise FileNotFoundError

        vocabulary_path = checkpoint_path + '.json'
        if not os.path.exists(vocabulary_path):
            vocabulary_path = os.path.join(os.path.dirname(checkpoint_path), 'vocab.json')
        if not os.path.exists(vocabulary_path):
            print('ERROR: could not load vocabulary')
            raise FileNotFoundError

        with open(vocabulary_path, 'r') as fp:
            vocab = json.load(fp)

        graph = tf.Graph()

        segments = {}

        #graph.as_default()

        with graph.as_default():
            saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
            sess = tf.Session()
            # with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
        
        self.graph = graph
        self.sess = sess
        self.vocab = vocab
        print(self.predict(np.zeros(self.sampling_rate), self.sampling_rate))
    
    def predict(self,wav_or_array,sr):
        vocab = self.vocab
        graph = self.graph
        n_batch = 256
        
        sound = None
        
        if isinstance(wav_or_array, str):
            sound, _ = audio_from_file(wav_or_array)
        elif isinstance(wav_or_array, np.ndarray):
            sound = wav_or_array
        else:
            return None
        
        if sr != self.sampling_rate:
            sound = librosa.resample(sound, sr , self.sampling_rate, res_type='kaiser_best')
        
        audio_duration = float(len(sound)) / self.sampling_rate
        
        # Transform audio
        out_of_frame_size = len(sound) % self.frame_size
        if out_of_frame_size > (0.2 * self.sampling_rate):
            sound = np.concatenate((sound,np.zeros(self.frame_size - out_of_frame_size)))
        else:
            sound = sound[:-out_of_frame_size]

        sound = np.reshape(sound,(-1, self.frame_size))
        if sound.shape[1] < self.sampling_rate:
            padding = np.zeros((sound.shape[0], self.sampling_rate - sound.shape[1]))
            sound = np.hstack((sound,padding))
        sound = np.reshape(sound,(-1))
        
        with self.graph.as_default():
            x = graph.get_tensor_by_name(vocab['x'])
            y = graph.get_tensor_by_name(vocab['y'])            
            init = graph.get_operation_by_name(vocab['init'])
            logits = graph.get_tensor_by_name(vocab['logits'])            
            ph_n_shuffle = graph.get_tensor_by_name(vocab['n_shuffle'])
            ph_n_repeat = graph.get_tensor_by_name(vocab['n_repeat'])
            ph_n_batch = graph.get_tensor_by_name(vocab['n_batch'])
            sr = vocab['sample_rate']
            
            input = audio_to_frames(sound, x.shape[1])
            labels = np.zeros((input.shape[0],), dtype=np.int32)
            self.sess.run(init, feed_dict = { x : input, y : labels, ph_n_shuffle : 1, ph_n_repeat : 1, ph_n_batch : n_batch })             
            count = 0
            n_total = input.shape[0]
            while True:
                try:
                    output = self.sess.run(logits)
                    output[:,0] += 0.21
                    output[:,1] -= 0.21
                    labels[count:count+output.shape[0]] = np.argmax(output, axis=1)
                    count += output.shape[0]
                except tf.errors.OutOfRangeError:                                                                               
                    break
        
        start_index = -1
        segs = []
        labels = labels.tolist()
        labels.append(0)
        for idx_ in range(0,len(labels)):
            
            if labels[idx_] == 1 and start_index == -1:
                start_index = idx_
            
            if labels[idx_] == 0:
                if start_index != -1:
                    segs.append([start_index,idx_])
                start_index = -1
            
        for i in range(0,len(segs)):
            
            segs[i][0] = segs[i][0] * self.frame_duration
            if segs[i][0] >= audio_duration:
                segs[i] = None
                continue
            segs[i][1] = min(segs[i][1] * self.frame_duration, audio_duration)
        
        segs = [seg for seg in segs if seg is not None]
        
        return segs
