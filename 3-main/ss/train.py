import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# 設置路徑
root_path = os.path.abspath(os.path.dirname(__file__))

# 超參數設置
learning_rate = 0.0001  # 降低學習率以提高穩定性
epochs = 200  # 增加訓練迭代次數
batch_size = 64  # 增加批量大小以提高梯度更新穩定性

def load_training_data():
    train_dataset = np.load(os.path.join(root_path, 'dataset', 'train.npz'))
    train_data = train_dataset['data']
    train_label = to_categorical(train_dataset['label'])
    # 標準化數據
    train_data = (train_data - np.mean(train_data, axis=0)) / (np.std(train_data, axis=0) + 1e-10)
    return train_data, train_label

def load_validation_data():
    valid_dataset = np.load(os.path.join(root_path, 'dataset', 'validation.npz'))
    valid_data = valid_dataset['data']
    valid_label = to_categorical(valid_dataset['label'])
    # 標準化數據
    valid_data = (valid_data - np.mean(valid_data, axis=0)) / (np.std(valid_data, axis=0) + 1e-10)
    return valid_data, valid_label

# 使用 MirroredStrategy 進行多 GPU 訓練
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(25,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),  # 增加 Dropout 避免過擬合
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # 假設分類數為 5
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    def train_model():
        train_data, train_label = load_training_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).shuffle(10000).batch(batch_size)

        # 訓練模型
        history = model.fit(
            train_dataset,
            epochs=epochs,
            verbose=1
        )

        # 保存模型架構與權重
        model.save(os.path.join(root_path, 'YOURMODEL.h5'))
        print(f"Model saved at {os.path.join(root_path, 'YOURMODEL.h5')}")

    def evaluate_model():
        valid_data, valid_label = load_validation_data()
        # 使用標準化後的數據進行評估
        loss, accuracy = model.evaluate(valid_data, valid_label, batch_size=batch_size, verbose=1)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    if __name__ == "__main__":
        train_model()
        accuracy = evaluate_model()
        bonus = int(accuracy * 100 // 5)
        print(f"Bonus points based on accuracy: {bonus}")
