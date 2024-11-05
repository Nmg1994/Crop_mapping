# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:18:01 2024

@author: Navid Mahdizadeh
"""

import rasterio
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, LSTM
import keras
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import sklearn
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

print('Optimal LSTM architecture applied to all study area')

Sentinel_2_images_masked_path = []

for num_img in range(22):
  Sentinel_2_images_masked_path.append(f'/home/navid94/Processed_Interval_{num_img}.tif')

Sen2_images_path = Sentinel_2_images_masked_path

def stack_and_normalize_satellite_images(images_path):
    sat_images = []
    
    for image_path in images_path:
        with rasterio.open(image_path) as src:
            each_image=[]
            
            for band_indice in range(1, 1 + src.count):
                band = src.read(band_indice)
                scaler = StandardScaler()
                normalized_band = scaler.fit_transform(band.reshape(-1, 1)).reshape(band.shape[0], band.shape[1])
                each_image.append(normalized_band)
            
            
            image = np.reshape(np.array(each_image), (np.array(each_image).shape[0], -1))
            sat_images.append(image)

    return np.stack(sat_images)


X_stacked = stack_and_normalize_satellite_images(Sen2_images_path)
print(X_stacked.shape)

transposed_X_stacked = np.transpose(X_stacked, (2, 1, 0))
print('transposed_X_stacked shape: ', transposed_X_stacked.shape)


# loading samples along with their corresponding labels # this is done to save some RAM due to the huge size of data
train_labels = np.load('/home/navid94/train_labels.npy')
val_labels = np.load('/home/navid94/val_labels.npy')
test_labels = np.load('/home/navid94/test_labels.npy')


train_samples = np.load('/home/navid94/train_samples.npy')
val_samples = np.load('/home/navid94/val_samples.npy')
test_samples = np.load('/home/navid94/test_samples.npy')

# Load the dictionary from the file
with open('/home/navid94/class_weights_dict_indices.pkl', 'rb') as f:
  class_weights_dict = pickle.load(f)

print(class_weights_dict)



Ground_truth_crop_data_path = '/home/navid94/CNN_LandC_Image.tif'

# Read the label image
with rasterio.open(Ground_truth_crop_data_path) as Ground_truth_crop_data_path_src:
  Ground_truth_image = Ground_truth_crop_data_path_src.read(1)  # a single band
  Ground_truth_image = Ground_truth_image.reshape(-1) - 1
  


def LSTM_classification(all_samples, True_CID_of_all_pixels, Features_training_LSTM, Features_val_LSTM, Features_testing_LSTM, landcover_training_LSTM, landcover_val_LSTM, landcover_testing_LSTM, time_steps, num_features, num_classes, Prediction_only = False):

  # Define early stopping callback
  early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

  # Defining the LSTM model
  LSTM_model = Sequential()
  LSTM_model.add(LSTM(units = 512, return_sequences= True, input_shape=(num_features, time_steps)))
  LSTM_model.add(Dropout(0.1))
  LSTM_model.add(LSTM(units = 128, return_sequences= True))
  LSTM_model.add(Dropout(0.1))
  LSTM_model.add(Flatten())
  LSTM_model.add(Dense(units = 512, activation = 'relu'))
  LSTM_model.add(Dense(num_classes, activation='softmax'))

  learning_rate = 0.001
  adam_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

  # Compile the model
  LSTM_model.compile(optimizer= adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])#'AUC', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), F1Score()])

  # Train the model
  LSTM_model.fit(Features_training_LSTM, landcover_training_LSTM, epochs=100, batch_size=32, validation_data=(Features_val_LSTM, landcover_val_LSTM),  callbacks=[early_stopping])


  LSTM_model.summary()


  #Testing Accuracy
  landcover_testing_predicted = LSTM_model.predict(Features_testing_LSTM)
  max_prob_indices = np.argmax(landcover_testing_predicted, axis=1)
  
  if Prediction_only == False:
    print("Recall score for test dataset: ", sklearn.metrics.recall_score(landcover_testing_LSTM, max_prob_indices, average = 'macro'))
    print("Precision score for test dataset: ", sklearn.metrics.precision_score(landcover_testing_LSTM, max_prob_indices, average = 'macro'))
    print("F1 score for test dataset: ", sklearn.metrics.f1_score(landcover_testing_LSTM, max_prob_indices, average = 'macro'))

    print(sklearn.metrics.classification_report(landcover_testing_LSTM, max_prob_indices))

    roc_auc_scores = []

    # Iterate over each class
    for class_index in range(13):
      # Extract the true labels and predicted probabilities for the current class
      y_true_class = np.array([1 if label == class_index else 0 for label in landcover_testing_LSTM])
      y_pred_class = np.array([1 if label == class_index else 0 for label in max_prob_indices])  # Predicted probabilities for the current class

      # Calculate the ROC-AUC score for the current class
      roc_auc_each_class = sklearn.metrics.roc_auc_score(y_true_class, y_pred_class)

      # Now, roc_auc_scores list contains the ROC-AUC score for each class
      print(f'ROC-AUC scores for class {class_index}:', roc_auc_each_class)

      # Append the ROC-AUC score to the list
      roc_auc_scores.append(roc_auc_each_class)
      
    max_prob_indices = max_prob_indices.reshape(-1,1)
    Cropland_predicted_LSTM = LSTM_model.predict(all_samples)
    max_prob_indices_Cropland_predicted_LSTM = np.argmax(Cropland_predicted_LSTM, axis=1)
    max_prob_indices_Cropland_predicted_LSTM = max_prob_indices_Cropland_predicted_LSTM.reshape(-1,1)

    conf_matrix_LSTM_test = confusion_matrix(landcover_testing_LSTM.reshape(-1,1), max_prob_indices)
    np.save('confusion_matrix_LSTM_test.npy', conf_matrix_LSTM_test)

    np.save('Crop_predicted_all_study_area.npy', max_prob_indices_Cropland_predicted_LSTM)
    np.save('True_Crops_ground_data.npy', True_CID_of_all_pixels)
    
    print('conf_matrix_LSTM_test: ', conf_matrix_LSTM_test)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix_LSTM_test, cmap='viridis', interpolation='nearest')

    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix of the LSTM Model on the Test Data')
    plt.xticks(ha='right')
    plt.yticks(ha='right')
    plt.savefig('Confusion_matrix_LSTM_test_data.jpeg')

    
  return LSTM_model

Cropland_predicted_samples_LSTM = LSTM_classification(all_samples = transposed_X_stacked, True_CID_of_all_pixels = np.array(Ground_truth_image).reshape(-1,1), Features_training_LSTM = train_samples, Features_val_LSTM = val_samples, Features_testing_LSTM = test_samples, landcover_training_LSTM = train_labels, landcover_val_LSTM = val_labels, landcover_testing_LSTM = test_labels, time_steps = train_samples.shape[2], num_features = train_samples.shape[1], num_classes = 13, Prediction_only = False)
print('Cropland_predicted_samples_LSTM: ', Cropland_predicted_samples_LSTM)
print('The run is finished!')
