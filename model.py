#Veriyi okuma ve işleme adımında kullanılacak olan kütüphaneler
import urllib
import itertools
import numpy as np
import pandas as pd
import random,os,glob
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from urllib.request import urlopen

# Warningleri kapatmak için
import warnings
warnings.filterwarnings("ignore")

# Model değerlendirme aşaması için
from sklearn.metrics import confusion_matrix, classification_report

#Model kurma için gerekli olan kütüphaneler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img