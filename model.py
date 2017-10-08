import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, ELU
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def saveImage(img, filename='image'):
    '''
    Utility function to save images for visualization.
    '''
    # opencv needs BGR colorspace
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def addDistortion(img):
    ''' 
    Method to add random brightness, shadows, and a shift in the horizon height.
    '''
    from skimage.exposure import adjust_gamma

    # add random brightness
    max_gamma = 0.1
    gamma = np.random.uniform(-1,1) * max_gamma + 1.0
    new_img = adjust_gamma(img, gamma)

    # add random shadow
    new_img = new_img.astype(float)
    h,w = new_img.shape[0:2]
    [top, bottom] = np.random.choice(np.arange(0,w,1), 2, replace=False)
    s = np.random.uniform(0.6,0.8)
    m = h / (bottom - top)
    b = -m * top
    for i in range(h):
        x = int((i - b) / m)
        if x < 0:
            x = 0
        if m > 0:
            new_img[i,:x,:] *= s
        else:
            new_img[i,x:,:] *= s

    # add random horizon shift
    h,w = new_img.shape[0:2]
    s = np.random.uniform(-15,15)
    M = np.float32([[1,0,0],[0,1,s]])
    new_img = cv2.warpAffine(new_img,M,(w,h))

    return new_img.astype(np.uint8)

def generator(data, batch_size=128):
    '''
    Method to generate data in parallel with training the network in batches.
    Random probablity to flip the image and angle as well as randomyl apply 
    distortion.
    '''
    num_samples = len(data)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset+batch_size]

            images = []
            steering_angles = []
            for sample in batch_samples:

                # read in images randomly from center, left, or right cameras
                camera = np.random.randint(3)
                img = cv2.cvtColor(cv2.imread(sample[camera]), cv2.COLOR_BGR2RGB)
                steering = float(sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.125 # this is a parameter to tune
                if camera == 1:
                    steering = steering + correction
                if camera == 2:
                    steering = steering - correction

                # randomly (50%) augment images and angles by flipping
                flip = np.random.randint(2)
                if flip == 0:
                    img = np.fliplr(img)
                    steering = -steering

                # randomly (50%) add distortion
                distort = np.random.randint(2)
                if distort == 0:
                    img = addDistortion(img)

                # add images and angles to data set
                images.append(img)
                steering_angles.append(steering)

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)

# append training samples
csv_files = []
csv_files.append('./track1/driving_log.csv')
csv_files.append('./track1-3laps-reverse/driving_log.csv')
csv_files.append('./track1-1lap-recovery/driving_log.csv')
csv_files.append('./track1-1lap-recovery-reverse/driving_log.csv')
csv_files.append('./track2-many-laps/driving_log.csv')
csv_files.append('./track2-recovery/driving_log.csv')
csv_files.append('./track2-extra/driving_log.csv')
csv_files.append('./track2-corner/driving_log.csv')

# load training samples
print("Reading csv files...")
samples = []
for csv_file in csv_files:
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        for line in reader:
            samples.append(line)
print("Read", len(samples), "samples.")

# split out 20% for validation
train_sample, validation_sample = train_test_split(samples, test_size=0.2)
print("train_sample:", len(train_sample), "validation_sample:",len(validation_sample))

# compile and train the model using the generator function
train_generator = generator(train_sample, batch_size=128)
validation_generator = generator(validation_sample, batch_size=128)

# add checkpoint funtion to save each epoch
checkpoint = ModelCheckpoint('model_{epoch:02d}.h5')

# the model and training
print("Training CNN...")
# CNN Model
# Adapted from Nvidia's Self Driving Car CNN model:
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
hp_drop = 0.5
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((40,20), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(hp_drop))
model.add(Dense(50))
model.add(Dropout(hp_drop))
model.add(Dense(10))
model.add(Dropout(hp_drop))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_sample)*5, 
    validation_data=validation_generator, nb_val_samples=len(validation_sample), nb_epoch=100,
    callbacks=[checkpoint])

############ Plot the model for visualization only #############
#from keras.utils.visualize_util import plot
#plot(model, to_file='model.png', show_shapes=True)
################################################################

# visualize a model summary
print(model.summary())

# save final model
model.save('model.h5')

# save a vidsualization of the training loss
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history.png')
 