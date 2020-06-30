import glob
import numpy as np
import math
import os
import re
import cv2 as cv
import keras
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import pickle

"""
This data generator is written for creating data for video classification model.
It is an keras.Sequence instance, so it can be easily used with keras.fit method
for learning deep models. The implementation is inspired by 
https://github.com/metal3d/keras-video-generators from which we borrowed some core functions,
simplified and removed many errors because of which our dataset did not work.
"""
class FrameGenerator(Sequence):
    def __init__(self, batch_size, number_of_frames, classes, target_shape, split_val_factor, nb_channel, video_path_format, augumentation=None, headers=True, typ="train", prepared_data=None):
        self.batch_size = batch_size
        self.rescale = 1./255.
        self.number_of_frames = number_of_frames
        self.classes = classes
        self.target_shape = target_shape
        self.headers = headers
        self.split_val_factor = split_val_factor
        self.nb_channel = nb_channel
        self.video_path_format = video_path_format
        self.augumentation = augumentation
        self._prepared_data = prepared_data

        self.files = []
        self.val_files = []
       
        self.typ = typ
        if self._prepared_data is not None:
            self.files = self._prepared_data
        else:
            self.load_file_lists(video_path_format, classes, split_val_factor)
        self.file_count = len(self.files)
        self.valid_count = len(self.val_files)
        self.class_nmbr = len(classes)

        self._framecount = {}
        self._c = 0
        self.ind = np.arange(self.file_count)
        self.preprared_aug = []
        self.on_epoch_end()
        print("Loaded",len(self.files),"files for",typ,"generator.")

    def load_file_lists(self, path_format, classes, split_val_factor):
        if split_val_factor is not None and split_val_factor > 0.0:
            for cls in classes:
                files = glob.glob(path_format.format(classname=cls))
                ind = np.arange(len(files))

                np.random.shuffle(ind)

                valid = int(len(files)*split_val_factor)
                per_list = np.random.permutation(ind)
                valid_list = per_list[:valid]
                train_list = per_list[valid:]

                self.files += [files[i] for i in train_list]
                self.val_files += [files[i] for i in valid_list]
                print("Loaded",len(train_list),"train",len(valid_list),"valid" ,"for class",cls,".")
        else:
            for cls in classes:
                t_f = glob.glob(path_format.format(classname=cls))
                self.files += t_f
                print("Loaded",len(t_f),self.typ,"files for class",cls,".")

    def get_valid(self):
        return self.__class__(self.batch_size, self.number_of_frames, self.classes, self.target_shape, None, self.nb_channel, self.video_path_format, None, typ="valid" ,prepared_data=self.val_files)
    def get_class_name(self,ind):
      return self.classes[ind]
    def write_files(self,p):
        with open(p, 'wb') as fp:
            pickle.dump(self.files, fp)
    def frame_count(self, cap, name, headers=True):
        
        if headers and name in self._framecount:
            return self._framecount[name]

        t = cap.get(cv.CAP_PROP_FRAME_COUNT)
        if not headers and t < 0:
            t = 0
            c = cv.VideoCapture(name)
            while True:
                grabbed, _ = c.read()
                if not grabbed:
                    break
                t += 1
        self._framecount[name] = t

        return t

    def next(self):
        n_e = self[self._c]
        self._c += 1

        if self._c == len(self):
            self._c = 0
            self.on_epoch_end()
        return n_e
    def on_epoch_end(self):
        np.random.shuffle(self.ind)

        if self.augumentation is not None:
            self.preprared_aug = []
            for i in range(self.file_count):
                self.preprared_aug.append(self.augumentation.get_random_transform(self.target_shape))
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def get_random_sample(self):
      while(True):
        video = np.random.choice(self.files)
        cl = self._get_class(video)
        label = np.zeros(self.class_nmbr)
        class_ind = self.classes.index(cl)
        if(class_ind != 22):
          continue
        label[class_ind] = 1.0
        
        frames = self._extract_frames(video, self.number_of_frames, self.target_shape, headers=self.headers)
        if frames is None:
          continue
        break
      return video, frames, label
    def __getitem__(self, ind):
        l = []
        imgs = []

        inds = self.ind[ind*self.batch_size:(ind+1)*self.batch_size]

        for i in inds:
            v = self.files[i]
            cl = self._get_class(v)

            lb = np.zeros(self.class_nmbr)
            class_ind = self.classes.index(cl)

            lb[class_ind] = 1.0

            f = self._extract_frames(v, self.number_of_frames, self.target_shape, headers=self.headers)

            if f is None:
                print("Could not get frames for",v)
                continue
            aug = None
            if self.augumentation is not None:
                f = [self.augumentation.apply_transform(fr,self.preprared_aug[i]) for fr in f]
            imgs.append(f)
            l.append(lb)

        l = np.array(l)
        imgs = np.array(imgs)

        return imgs, l

    def _get_class(self, video):
        video = os.path.realpath(video)
        pattern = os.path.realpath(self.video_path_format)

        pattern = re.escape(pattern)

        pattern = pattern.replace('\\*', '.*')

        pattern = pattern.replace('\\{classname\\}', '(.*?)')

        return re.findall(pattern, video)[0]
        

    def _extract_frames(self, video, number_of_frames, shape, headers=True):
        cap = cv.VideoCapture(video)
        total_frames = self.frame_count(cap, video)
        frame_step = math.floor(total_frames/number_of_frames/2)

        frame_step = max(1, frame_step)
        frames = []
        frame_i = 0

        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            frame_i += 1
            if frame_i % frame_step == 0:
                frame = cv.resize(frame, shape)

                if self.nb_channel == 3:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                else:
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                frame = img_to_array(frame) * self.rescale

                frames.append(frame)

            if len(frames) == number_of_frames:
                break

        cap.release()

        if headers and len(frames) != number_of_frames:
            print("Invalid frame count for video",video,". Requested:",number_of_frames,"Obtained:",len(frames))
            return self._extract_frames(video,number_of_frames,shape,headers=False)
        if not headers and len(frames) != number_of_frames:
            print("No other option for",video)
            return None
        return np.array(frames)


