import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# 虛構的文本數據集
texts = ["I love Python programming!",
         "TensorFlow is an amazing tool.",
         "NLP is fascinating.",
         "I hate deadlines.",
         "This is just a sample text.",
         "I am neutral about this."]

# 對文本進行標記並轉換為序列
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
max_sequence_length = max([len(seq) for seq in sequences])

# 填充序列以確保它們具有相同的長度
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

# 虛構的情感標籤
labels = np.array([1, 1, 1, 0, 0, 2])  # 1: 正面, 0: 負面, 2: 中性

# 將數據集拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# 建立情感分類模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(len(word_index) + 1, 32))  # 移除 input_length 參數
model.add(tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(3, activation='softmax'))  # 3個類別: 正面、負面、中性

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# 測試模型
while True:
    text = input("Enter a text to classify its sentiment (or 'exit' to quit): ")
    if text.lower() == 'exit':
        break
    else:
        # 將輸入的文本轉換為序列
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_sequence_length)
        # 進行情感分類
        result = model.predict(padded_sequence)
        # 解析結果
        if np.argmax(result) == 0:
            print("Negative sentiment")
        elif np.argmax(result) == 1:
            print("Positive sentiment")
        else:
            print("Neutral sentiment")
