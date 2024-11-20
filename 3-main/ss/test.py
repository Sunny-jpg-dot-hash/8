from tensorflow import keras
import numpy as np
import os
from keras.utils import to_categorical

def main():

    root_path = os.path.abspath(os.path.dirname(__file__))

    #放入模型.h5
    model = keras.models.load_model(root_path + '/YOURMODEL.h5')

    #放入test.npz
    test = np.load(root_path + '/test2_1.npz')

    data = test['data']
    test_label = test['label']


    test_label = to_categorical(test_label, num_classes=5)
    
    predictions = model.predict(data)
    predicted_classes = np.argmax(predictions, axis=1)

    true_labels = np.argmax(test_label, axis=1)
    accuracy = np.mean(true_labels == predicted_classes)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()

#group two