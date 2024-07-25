# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'alzheimer-mri-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F2029496%2F3364939%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240717%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240717T083804Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D4df80461218989122e53217c4dbc997c0796fd163a9cffcda8480e6db2b3ef02265437d735a5f4559e37ace437a04eca2b301dd5b2d8fa7f49e7ccd2003cf9efd642f3ef2993b396a056933a221373b94cf1db35db2e22b0b6249923cd294d3e021898533537316bba9099a1f2b8b328cf5d7d5939f0177b7afd59d39cbca36e9b617575114519168558f8da3c91df22050a5c47efe76263aebb96d270ee612ca65eb987b73a1a523765b57e75c21e3f79640499f0f0b70551e3adf36cb533a66c05a6257e8450ead5509b77749244f220388eed0e2b7daca87a78154fc367d1ea5307f604f20a4c957c62cba39f8b9f362b24d7ac959520e9c5cbc8f1a64271'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')
# import system libs
import os
import time
import shutil
import pathlib
import itertools

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('Modules loaded successfuly!!!')
import dask.array as da

# Create a large Dask array
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# Perform a computation (e.g., mean)
result = x.mean().compute()

print(result)
import cupy as cp

# Create a large CuPy array
x = cp.random.random((10000, 10000))

# Perform a computation (e.g., mean)
result = cp.mean(x)

print(result)
# Enable Intel oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
# Generate data paths with labels
def define_paths(data_dir):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    return filepaths, labels


# Concatenate data paths with labels into one dataframe ( to later be fitted into the model )
def define_df(files, classes):
    Fseries = pd.Series(files, name= 'filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis= 1)

# Split dataframe to train, valid, and test
def split_data(data_dir):
    # train dataframe
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    strat = df['labels']
    train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 323, stratify= strat)

    # valid and test dataframe
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df,  train_size= 0.5, shuffle= True, random_state= 323, stratify= strat)

    return train_df, valid_df, test_df

print("This code cell run successfully!!!")
def create_gens (train_df, valid_df, test_df, batch_size):

    # define model parameters
    img_size = (224, 224)
    channels = 3
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)

    # Recommended : use custom function for test data batch size, else we can use normal batch size.
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    # This function which will be used in image data generator for data augmentation, it just take the image and return it again.
    def scalar(img):
        # Example of augmentation: rescaling, rotating, and flipping
        img = tf.image.resize(img, [224, 224])  # Resize to 224x224
        img = tf.image.random_flip_left_right(img)  # Random horizontal flip
        img = tf.image.random_brightness(img, max_delta=0.1)  # Random brightness adjustment
        return img / 255.0  # Normalize to [0, 1]

    tr_gen = ImageDataGenerator(preprocessing_function= scalar)
    ts_gen = ImageDataGenerator(preprocessing_function= scalar)

    train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    # Note: we will use custom test_batch_size, and make shuffle= false
    test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= False, batch_size= test_batch_size)

    return train_gen, valid_gen, test_gen
print("This code cell run successfully!!!")
def show_images(gen):
    '''
    This function take the data generator and show sample of the images
    '''

    # return classes , images to be displayed
    g_dict = gen.class_indices        # defines dictionary {'class': index}
    classes = list(g_dict.keys())     # defines list of dictionary's kays (classes), classes names : string
    images, labels = next(gen)        # get a batch size samples from the generator

    # calculate number of displayed samples
    length = len(labels)        # length of batch size
    sample = min(length, 25)    # check if sample less than 25 images

    plt.figure(figsize= (20, 20))

    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i]       # scales data to range (0 - 255)
       # Normalize image if necessary
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image, 0, 1)
        elif image.dtype == np.uint8:
            image = np.clip(image, 0, 255)
        else:
            raise ValueError(f"Unexpected image data type: {image.dtype}")
        # plt.imshow(image)
        plt.imshow(image)
        index = np.argmax(labels[i])  # get image index
        class_name = classes[index]   # get class of image
        plt.title(class_name, color= 'blue', fontsize= 12)
        plt.axis('off')
    plt.show()
print("This code cell run successfully!!!")
def plot_training(hist):
    '''
    This function take training model and plot history of accuracy and losses with the best epoch in both of them.
    '''

    # Define needed variables
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    # Plot training history
    plt.figure(figsize= (10, 4))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.show()
print("This code cell run successfully!!!")
def plot_confusion_matrix(cm, classes, normalize= False, title= 'Confusion Matrix', cmap= plt.cm.Blues):
	'''
	This function plot confusion matrix method from sklearn package.
	'''

	plt.figure(figsize= (10, 10))
	plt.imshow(cm, interpolation= 'nearest', cmap= cmap)
	plt.title(title)
	plt.colorbar()

	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation= 45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis= 1)[:, np.newaxis]
		print('Normalized Confusion Matrix')

	else:
		print('Confusion Matrix, Without Normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

	plt.tight_layout()
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
print("This code cell run successfully!!!")
# To compare the evaluation results through graph plotting
def plot_comparison(training_time_no_hpc, training_time_hpc, train_eval_no_hpc, valid_eval_no_hpc, test_eval_no_hpc, train_eval_hpc, valid_eval_hpc, test_eval_hpc):
    labels = ['Training', 'Validation', 'Testing']

    # Training times comparison
    times = [training_time_no_hpc, training_time_hpc]
    plt.figure(figsize=(10, 6))
    plt.bar(['Without HPC', 'With HPC'], times, color=['red', 'blue'])
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.show()

    # Evaluation metrics comparison
    metrics = ['Loss', 'Accuracy']
    evaluations_no_hpc = [train_eval_no_hpc, valid_eval_no_hpc, test_eval_no_hpc]
    evaluations_hpc = [train_eval_hpc, valid_eval_hpc, test_eval_hpc]

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        plt.plot(labels, [eval[i] for eval in evaluations_no_hpc], marker='o', label='Without HPC', color='red')
        plt.plot(labels, [eval[i] for eval in evaluations_hpc], marker='o', label='With HPC', color='blue')
        plt.title(f'{metric} Comparison')
        plt.xlabel('Dataset')
        plt.ylabel(metric)
        plt.legend()
        plt.show()
print("This code cell run successfully!!!")
data_dir = '/kaggle/input/alzheimer-mri-dataset/Dataset'

try:
    # Get splitted data
    train_df, valid_df, test_df = split_data(data_dir)

    # Get Generators
    batch_size = 40
    train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, batch_size)

except:
    print('Invalid Input')
  # Create Model Structure
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer

# create pre-trained model (you can built on pretrained model such as :  efficientnet, VGG , Resnet )
# we will use efficientnetb3 from EfficientNet family.
base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max') #(224,224,3)

model = Sequential([
    base_model,
    BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
    Dropout(rate= 0.45, seed= 123),
    Dense(class_count, activation= 'softmax')
])

model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

model.summary()
batch_size = 40   # set batch size for training
epochs = 40   # number of all epochs in training
patience = 1   #number of epochs to wait to adjust lr if monitored value does not improve
stop_patience = 3   # number of epochs to wait before stopping training if monitored value does not improve
threshold = 0.9   # if train accuracy is < threshold adjust monitor accuracy, else monitor validation loss
factor = 0.5   # factor to reduce lr by
ask_epoch = 5   # number of epochs to run before asking if you want to halt training
batches = int(np.ceil(len(train_gen.labels) / batch_size))    # number of training batch to run per epoch

callbacks = [MyCallback(model= model, patience= patience, stop_patience= stop_patience, threshold= threshold,
            factor= factor, batches= batches, epochs= epochs, ask_epoch= ask_epoch )]
# Train and evaluate without HPC optimizations
print("Training without HPC optimizations:")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
callbacks = MyCallback(model, patience, stop_patience, threshold, factor, batches, epochs, ask_epoch)

start_time = time.time()
history_no_hpc = model.fit(train_gen, validation_data=valid_gen, epochs=epochs, callbacks=[callbacks])
end_time = time.time()
training_time_no_hpc = end_time - start_time

train_eval_no_hpc = model.evaluate(train_gen)
valid_eval_no_hpc = model.evaluate(valid_gen)
test_eval_no_hpc = model.evaluate(test_gen)

print("Train Loss: ", train_eval_no_hpc[0])
print("Train Accuracy: ", train_eval_no_hpc[1])
print('-' * 20)
print("Validation Loss: ", valid_eval_no_hpc[0])
print("Validation Accuracy: ", valid_eval_no_hpc[1])
print('-' * 20)
print("Test Loss: ", test_eval_no_hpc[0])
print("Test Accuracy: ", test_eval_no_hpc[1])
# Train and evaluate with HPC optimizations
print("Training with HPC optimizations:")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN
callbacks = MyCallback(model, patience, stop_patience, threshold, factor, batches, epochs, ask_epoch)

start_time = time.time()
history_hpc = model.fit(train_gen, validation_data=valid_gen, epochs=epochs, callbacks=[callbacks])
end_time = time.time()
training_time_hpc = end_time - start_time

train_eval_hpc = model.evaluate(train_gen)
valid_eval_hpc = model.evaluate(valid_gen)
test_eval_hpc = model.evaluate(test_gen)

print("Train Loss: ", train_eval_hpc[0])
print("Train Accuracy: ", train_eval_hpc[1])
print('-' * 20)
print("Validation Loss: ", valid_eval_hpc[0])
print("Validation Accuracy: ", valid_eval_hpc[1])
print('-' * 20)
print("Test Loss: ", test_eval_hpc[0])
print("Test Accuracy: ", test_eval_hpc[1])
# Compare training times
print(f"Training time without HPC optimizations: {training_time_no_hpc:.2f} seconds")
print(f"Training time with HPC optimizations: {training_time_hpc:.2f} seconds")
# Compare training times
print(f"Training time without HPC optimizations: {training_time_no_hpc:.2f} seconds")
print(f"Training time with HPC optimizations: {training_time_hpc:.2f} seconds")
print("Training and Validation Graph for using HPC!!!\n")
plot_training(history_hpc)
# Predict and evaluate
test_gen.reset()
preds = model.predict(test_gen, steps=test_gen.samples // test_gen.batch_size + 1, verbose=1)
pred_labels = np.argmax(preds, axis=1)
true_labels = test_gen.classes
# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
classes = list(test_gen.class_indices.keys())
plot_confusion_matrix(cm, classes)

# Classification report
report = classification_report(true_labels, pred_labels, target_names=classes)
print(report)
# Call the function to plot comparisons
plot_comparison(training_time_no_hpc, training_time_hpc, train_eval_no_hpc, valid_eval_no_hpc, test_eval_no_hpc, train_eval_hpc, valid_eval_hpc, test_eval_hpc)
