# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: Navid Mahdizadeh Gharakhanlou

Searching the optimized initial LSTM architecture
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, LSTM
import sklearn
import sklearn.metrics
from keras.callbacks import EarlyStopping
import xlsxwriter
from itertools import product
import pickle

print('Searching the optimized initial LSTM architecture')

# Loading samples along with their corresponding labels # this is done to save some RAM due to the huge size of data
train_labels = np.load('/home/navid94/train_labels.npy')
val_labels = np.load('/home/navid94/val_labels.npy')
test_labels = np.load('/home/navid94/test_labels.npy')


train_samples = np.load('/home/navid94/train_samples.npy')
val_samples = np.load('/home/navid94/val_samples.npy')
test_samples = np.load('/home/navid94/test_samples.npy')

# Loading the class weight dictionary from the file
with open('/home/navid94/class_weights_dict.pkl', 'rb') as f:
  class_weights_dict = pickle.load(f)

print(class_weights_dict)

# Defining hyperparameters
architectures = {'LSTM': LSTM}

layer_counts = [1, 2, 3, 4]
cells_per_layer = [32, 64, 128, 256, 512]
dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Function to create and compile a neural network
def create_network(architecture, num_layers, num_cells, dropout_rate, num_classes):
    model = Sequential()
    for _ in range(num_layers):
        model.add(architecture(units=num_cells, return_sequences=True))
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(units = 512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

workbook = xlsxwriter.Workbook("LSTM_architecture.xlsx")
worksheet = workbook.add_worksheet("LSTM")
worksheet.write('A1', 'num_layers')
worksheet.write('B1', 'num_cells')
worksheet.write('C1', 'dropout_rate')
worksheet.write('D1', 'Validation_accuracy')

# Training and evaluating the networks using training and validation datasets, respectively
for architecture_name, architecture in architectures.items():
  row_index = 2

  for num_layers, num_cells, dropout_rate_val in product(layer_counts, cells_per_layer, dropout_rate):
    # Creating and compiling a LSTM model
    model = create_network(architecture, num_layers, num_cells, dropout_rate_val, num_classes = 13)

    # Printing architecture and hyperparameters
    print(f"Architecture: {architecture_name}, Number of Layers: {num_layers}, Number of Cells per Layer: {num_cells}, Dropout rate: {dropout_rate_val}")

    model.fit(train_samples, train_labels, epochs=100, class_weight=class_weights_dict, batch_size=32, validation_data=(val_samples, val_labels),  callbacks=[early_stopping])


    landcover_validation_predicted = model.predict(val_samples)
    validation_max_prob_indices = np.argmax(landcover_validation_predicted, axis=1)

    val_evaluation = sklearn.metrics.f1_score(val_labels, validation_max_prob_indices, average = 'macro') 
     

    print("Recall score for test dataset: ", sklearn.metrics.recall_score(val_labels, validation_max_prob_indices, average = 'macro'))
    print("Precision score for test dataset: ", sklearn.metrics.precision_score(val_labels, validation_max_prob_indices, average = 'macro'))
    print("F1 score for test dataset: ", sklearn.metrics.f1_score(val_labels, validation_max_prob_indices, average = 'macro'))
    print(sklearn.metrics.classification_report(val_labels, validation_max_prob_indices))
    
    worksheet.write('A' + str(row_index), num_layers)
    worksheet.write('B' + str(row_index), num_cells)
    worksheet.write('C' + str(row_index), dropout_rate_val)
    worksheet.write('D' + str(row_index), val_evaluation)
    row_index += 1

workbook.close()
