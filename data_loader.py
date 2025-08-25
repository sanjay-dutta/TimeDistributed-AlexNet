import os
import cv2 as cv
import numpy as np
from glob import glob
from random import shuffle
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split

IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
SEQUENCE_LENGTH = 25
BATCH_SIZE = 8
NUM_CLASSES = 10
CLASSES_LIST = [
    'Arguing', 'Eating_in_classroom', 'Explaining_the_Subject', 'HandRaise',
    'Holding_Book', 'Holding_Mobile_Phone', 'Reading_Book',
    'Sitting_on_Desk', 'Writing_On_Board', 'Writting_on_Textbook'
]

class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.video_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(batch_paths, batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.video_paths, self.labels))
            shuffle(temp)
            self.video_paths, self.labels = zip(*temp)

    def __data_generation(self, batch_paths, batch_labels):
        X, y = [], []
        for path, label in zip(batch_paths, batch_labels):
            cap = cv.VideoCapture(path)
            total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            skip = max(int(total / SEQUENCE_LENGTH), 1)
            frames = []
            for i in range(SEQUENCE_LENGTH):
                cap.set(cv.CAP_PROP_POS_FRAMES, i * skip)
                success, frame = cap.read()
                if not success:
                    break
                frame = cv.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)) / 255.0
                frames.append(frame)
            cap.release()
            if len(frames) == SEQUENCE_LENGTH:
                X.append(frames)
                label_cat = to_categorical(label, NUM_CLASSES)
                label_seq = np.repeat(label_cat[np.newaxis, :], SEQUENCE_LENGTH, axis=0)
                y.append(label_seq)
        return np.array(X), np.array(y)

def get_generators(dataset_dir):
    video_paths, labels = [], []
    for idx, class_name in enumerate(CLASSES_LIST):
        class_dir = os.path.join(dataset_dir, class_name)
        videos = glob(os.path.join(class_dir, "*"))
        video_paths.extend(videos)
        labels.extend([idx] * len(videos))

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=27
    )

    train_gen = VideoDataGenerator(train_paths, train_labels, batch_size=BATCH_SIZE)
    test_gen = VideoDataGenerator(test_paths, test_labels, batch_size=BATCH_SIZE, shuffle=False)
    return train_gen, test_gen
