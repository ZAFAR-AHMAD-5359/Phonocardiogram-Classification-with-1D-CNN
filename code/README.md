import os
import numpy as np
import pandas as pd
import librosa
import librosa.feature
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set the paths to your data directories and labels files
train_data_dir = '/content/drive/MyDrive/Heart-Multiclass/DeepSignal/TrainData'
train_labels_file = '/content/drive/MyDrive/Heart-Multiclass/DeepSignal/TrainLabels.xlsx'
test_data_dir = '/content/drive/MyDrive/Heart-Multiclass/DeepSignal/TestData'
test_labels_file = '/content/drive/MyDrive/Heart-Multiclass/DeepSignal/TestLabel.xlsx'

def specificity(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    tn = conf_matrix.sum() - (conf_matrix.sum(axis=1) + fp)
    specificity = tn / (tn+fp)
    return np.mean(specificity)

# Modify the load_data function
def load_data(data_dir, labels_file):
    labels_df = pd.read_excel(labels_file)
    data = []
    labels = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_dir, file_name)
            signal, sr = librosa.load(file_path, sr=None, duration=3.0)

            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

            matching_rows = labels_df.loc[labels_df['Name'] == file_name]
            if len(matching_rows) > 0:
                label = matching_rows['Label'].values[0]
                data.append(mfccs)
                labels.append(label)

    return data, labels

# Load training and test data
train_data, train_labels = load_data(train_data_dir, train_labels_file)
test_data, test_labels = load_data(test_data_dir, test_labels_file)

# Convert MFCCs to numpy arrays
train_data_mfccs = np.array(train_data)
test_data_mfccs = np.array(test_data)

# Reshape the data for Conv1D
train_data_mfccs = np.expand_dims(train_data_mfccs, axis=3)  # Add an extra dimension for Conv1D
test_data_mfccs = np.expand_dims(test_data_mfccs, axis=3)

# Update the input_shape to match the shape of MFCCs
input_shape = train_data_mfccs.shape[1:]  # Shape without the number of samples

# Convert labels to numeric format
label_encoder = LabelEncoder()
encoded_train_labels = label_encoder.fit_transform(train_labels)
encoded_test_labels = label_encoder.transform(test_labels)

num_classes = len(label_encoder.classes_)

# Determine the maximum length of signals
max_length = max(max(len(signal) for signal in train_data), max(len(signal) for signal in test_data))

# Prepare the data
train_data = np.array([np.pad(signal, (0, max_length - len(signal)), constant_values=0) for signal in train_data])
test_data = np.array([np.pad(signal, (0, max_length - len(signal)), constant_values=0) for signal in test_data])

train_data = np.expand_dims(train_data, axis=2)
test_data = np.expand_dims(test_data, axis=2)

train_labels = to_categorical(encoded_train_labels, num_classes=num_classes)
test_labels = to_categorical(encoded_test_labels, num_classes=num_classes)

# Prepare cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize results
train_acc_results = []
val_acc_results = []
test_acc_results = []
sensitivity_results = []
specificity_results = []
f1_score_results = []

# Loop over each fold
for fold, (train_index, test_index) in enumerate(kf.split(train_data_mfccs, encoded_train_labels)):
    X_train, X_test = train_data_mfccs[train_index], train_data_mfccs[test_index]
    y_train, y_test = train_labels[train_index], train_labels[test_index]

    model = Sequential()
    input_shape = X_train.shape[1:]  # Shape without the number of samples
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))  # Add another Conv1D layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))  # Add another Conv1D layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))  # Add another Conv1D layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))  # Add another Conv1D layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))  # Add another Conv1D layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))  # Add another Conv1D layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))  # Add another Conv1D layer
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    learning_rate = 0.00001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Define early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=8, epochs=1000, callbacks=[early_stopping], verbose=1)

    # Get the training and validation accuracy from history
    final_training_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]

    # Get the training and test accuracy from history
    final_training_accuracy = history.history['accuracy'][-1]
    final_test_accuracy = history.history['val_accuracy'][-1]  # renamed from final_val_accuracy

    # Save the results
    train_acc_results.append(final_training_accuracy)
    val_acc_results.append(final_val_accuracy)

    # Save the results
    train_acc_results.append(final_training_accuracy)
    test_acc_results.append(final_test_accuracy)  # renamed from val_acc_results


    # Plot training and validation accuracy
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy for Fold {fold+1}')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for Fold {fold+1}')
    plt.legend()


    # Plot training, validation and test accuracy
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(test_acc_results, label='Test Accuracy')  # Added this line
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training, Validation, and Test Accuracy for Fold {fold+1}')
    plt.legend()

    plt.tight_layout()
    plt.show()


    # Compute and plot confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for Fold {fold+1}')
    plt.show()

    # Compute and plot confusion matrix for test data
    y_pred = model.predict(test_data_mfccs)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(test_labels, axis=1)
    confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for Test Data (Fold {fold+1})')
    plt.show()

    # Compute Sensitivity, Specificity, F1 Score
    sensitivity = recall_score(y_test_classes, y_pred_classes, average='micro')
    specificity_score = specificity(y_test_classes, y_pred_classes)  # renamed the variable
    f1 = f1_score(y_test_classes, y_pred_classes, average='micro')

    sensitivity_results.append(sensitivity)
    specificity_results.append(specificity_score)  # use the renamed variable
    f1_score_results.append(f1)



    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for this Fold')
    plt.show()

    # Compute Sensitivity, Specificity, F1 Score for final test data
    sensitivity = recall_score(y_test_classes, y_pred_classes, average='micro')
    specificity_score = specificity(y_test_classes, y_pred_classes)
    f1 = f1_score(y_test_classes, y_pred_classes, average='micro')

    test_accuracy = accuracy_score(y_test_classes, y_pred_classes)
    test_acc_results.append(test_accuracy)

    print('Final Test Accuracy:', test_accuracy)
    print('Final Test Sensitivity:', sensitivity)
    print('Final Test Specificity:', specificity_score)
    print('Final Test F1 Score:', f1)

    print('Final training accuracy per fold:', train_acc_results)
    print('Final validation accuracy per fold:', val_acc_results)
    print('Final test accuracy per fold:', test_acc_results)
    print('Average final training accuracy:', np.mean(train_acc_results))
    print('Average final validation accuracy:', np.mean(val_acc_results))
    print('Average final test accuracy:', np.mean(test_acc_results))

    print('Final sensitivity per fold:', sensitivity_results)
    print('Final specificity per fold:', specificity_results)
    print('Final F1 score per fold:', f1_score_results)
    print('Average final sensitivity:', np.mean(sensitivity_results))
    print('Average final specificity:', np.mean(specificity_results))
    print('Average final F1 score:', np.mean(f1_score_results))
