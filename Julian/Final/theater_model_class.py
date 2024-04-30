import tensorflow as tf
import PIL
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dropout, Dense, Softmax)
from tensorflow.keras.applications import mobilenet as _mobilenet
import random
import os
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image
import re
import heapq
import pickle

def clean_filename(filename):
    """ Removes any special characters from filename except alphanumeric, spaces, dots, and hyphens. """
    return re.sub(r'[^\w\s.-]', '', filename)

def load_image(img_file, target_size=(224,224)):
    """ Loads and processes an image to be ready for model prediction. """
    X = np.zeros((1, *target_size, 3))
    X[0, ] = np.asarray(tf.keras.preprocessing.image.load_img(img_file, target_size=target_size))
    X = tf.keras.applications.mobilenet.preprocess_input(X)
    return X

def ensure_folder_exists(folder):
    """ Creates the folder if it does not exist. """
    if not os.path.exists(folder):
        os.makedirs(folder)

class FilmProcessor:
    def __init__(self, folder_path, model):
        self.folder_path = folder_path
        self.model = model
        file_path = r"D:\sp\SpainTheatre.pkl"
        with open(file_path, 'rb') as file:
            self.film_features = pickle.load(file)

    def test(self,q):
        return q

    def process_all_mp4_files(self):
        """ Processes all mp4 files in the specified directory, cleaning filenames and collecting file info. """
        mp4_files = []
        for file in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file)
            if os.path.isfile(file_path) and file.endswith(".mp4"):
                cleaned_file_name = clean_filename(file)
                mp4_files.append({
                    'original_name': file,
                    'cleaned_name': cleaned_file_name,
                    'full_path': file_path
                })
        return mp4_files

    def processFilmByFrames(self, filmName, filmPath, interval):
        """ Processes a film by extracting frames at a given interval and computing their features. """
        features = []
        film = VideoFileClip(filmPath)
        nrFrames = int(film.duration / interval)
        ensure_folder_exists('/content/drive/MyDrive/FinalWork/spanish stage play/Frames')
        for i in range(nrFrames):
            s = i * interval
            frame = film.get_frame(s)
            frame_image = Image.fromarray(frame, 'RGB')
            temp_frame_path = f'Frames/{filmName}_{i}.jpg'
            frame_image.save(temp_frame_path)
            im = load_image(temp_frame_path)
            f = self.model.predict(im)[0]
            features.append({'film': filmPath, 'second': s, 'features': f, 'frame_path': temp_frame_path})
        return features

    def processImage(self, imagePath):
        """ Processes an image to extract features using the loaded model. """
        im = load_image(imagePath)
        f = self.model.predict(im)[0]
        return f

    def findTopFramesByFeatures(self, queryFeatures, top_n=400):
        """ Finds the top frames that match the query features based on Euclidean distance. """
        top_frames_heap = []
        import heapq
        for f in self.film_features:
            dist = np.linalg.norm(f['features'] - queryFeatures)
            if len(top_frames_heap) < top_n:
                heapq.heappush(top_frames_heap, (-dist, f['film'], f['second'], f['frame_path']))
            else:
                heapq.heappushpop(top_frames_heap, (-dist, f['film'], f['second'], f['frame_path']))
        top_frames_sorted = sorted(top_frames_heap, key=lambda x: x[0], reverse=True)
        return [(frame[1], frame[2], frame[3]) for frame in top_frames_sorted]
