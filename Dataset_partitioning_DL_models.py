# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: Navid Mahdizadeh

Dataset partitioning in region-based DL models
"""

import rasterio
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle

Sentinel_2_images_masked_path = []

for num_img in range(43):
  Sentinel_2_images_masked_path.append(f'/home/navid94/Sentinel2_Interval_{num_img}_CNN_masked.tif')

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


Ground_truth_crop_data_path = '/home/navid94/CNN_LandC_Image.tif'

# Read the label image
with rasterio.open(Ground_truth_crop_data_path) as Ground_truth_crop_data_path_src:
  Ground_truth_image = Ground_truth_crop_data_path_src.read(1)  # a single band
  Ground_truth_image = Ground_truth_image.reshape(-1) - 1

  print(np.unique(Ground_truth_image))
  # Get the unique values and their counts
  unique_values, counts = np.unique(Ground_truth_image, return_counts=True)

  # Initialize lists to store pixel indices for each dataset split
  train_indices = []
  val_indices = []
  test_indices = []

  
  # Print unique values and their counts
  for value_class, count_each_class in zip(unique_values, counts):
    print(f"Value: {value_class}, Count: {count_each_class}")

    indices_specific_value = np.where(Ground_truth_image == value_class)[0]

    if value_class == 0:
      count = math.ceil(0.03 * count_each_class)
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 1:
      count = math.ceil(0.1 * count_each_class)
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 2:
      count = math.ceil(0.05 * count_each_class)
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 3:
      count = math.ceil(0.5 * count_each_class)
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 4:
      count = math.ceil(0.01 * count_each_class)
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 5:
      count = math.ceil(0.05 * count_each_class)
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 6:
      count = math.ceil(0.5 * count_each_class)
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 7:
      count = count_each_class
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 8:
      count = count_each_class
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 9:
      count = count_each_class
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 10:
      count = math.ceil(0.1 * count_each_class)
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 11:
      count = math.ceil(0.1 * count_each_class)
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
    elif value_class == 12:
      count = math.ceil(0.05 * count_each_class)
      indices = np.random.choice(indices_specific_value, size=count, replace=False)
      
    
    train_pixels = int(count * 0.6)
    val_pixels = int(count * 0.2)
    test_pixels = count - train_pixels - val_pixels
    
    
    # Split the indices into training, validation, and testing sets
    train_indices.append(indices[:train_pixels])
    val_indices.append(indices[train_pixels:train_pixels + val_pixels])
    test_indices.append(indices[train_pixels + val_pixels:])
    
    if value_class == 0:
      train_labels= Ground_truth_image[indices[:train_pixels]]
      val_labels = Ground_truth_image[indices[train_pixels:train_pixels + val_pixels]]
      test_labels = Ground_truth_image[indices[train_pixels + val_pixels:]]

      train_samples = transposed_X_stacked[indices[:train_pixels]]
      val_samples = transposed_X_stacked[indices[train_pixels:train_pixels + val_pixels]]
      test_samples = transposed_X_stacked[indices[train_pixels + val_pixels:]]
    else:
      
      train_labels= np.concatenate([train_labels, Ground_truth_image[indices[:train_pixels]]], axis=0)
      val_labels = np.concatenate([val_labels, Ground_truth_image[indices[train_pixels:train_pixels + val_pixels]]], axis = 0)
      test_labels = np.concatenate([test_labels, Ground_truth_image[indices[train_pixels + val_pixels:]]] , axis = 0)

      train_samples = np.concatenate([train_samples, transposed_X_stacked[indices[:train_pixels]]], axis=0)
      val_samples = np.concatenate([val_samples, transposed_X_stacked[indices[train_pixels:train_pixels + val_pixels]]], axis=0)
      test_samples = np.concatenate([test_samples, transposed_X_stacked[indices[train_pixels + val_pixels:]]], axis=0)

  np.save('train_indices.npy', np.array(np.concatenate(train_indices)).reshape(-1))
  np.save('val_indices.npy', np.array(np.concatenate(val_indices)).reshape(-1))
  np.save('test_indices.npy', np.array(np.concatenate(test_indices)).reshape(-1))

  np.save('train_labels.npy', train_labels)
  np.save('val_labels.npy', val_labels)
  np.save('test_labels.npy', test_labels)
  
  np.save('train_samples.npy', train_samples)
  np.save('val_samples.npy', val_samples)
  np.save('test_samples.npy', test_samples)

  # Compute class weights based on the distribution of classes in the training dataset
  class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels.reshape(-1)), y=train_labels.reshape(-1))

  # Convert class weights to a dictionary format
  class_weights_dict = dict(zip(np.unique(train_labels.reshape(-1)), class_weights))

  with open('class_weights_dict_indices.pkl', 'wb') as f:
    pickle.dump(class_weights_dict, f)
