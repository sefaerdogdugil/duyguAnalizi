from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# IMDb veri setini yükleme (25.000 eğitim, 25.000 test)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)  # En sık geçen 10,000 kelime.

# Padding işlemi (tüm dizilerin uzunluğunu sabitleme)
max_length = 200  # Tüm yorumları 200 kelimeye kadar kısaltıyoruz veya dolduruyor.
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=32, input_length=max_length))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))  # LSTM katmanı
model.add(Dense(1, activation='sigmoid'))  # İkili sınıflandırma için sigmoid aktivasyon fonksiyonu

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=7, batch_size=128, validation_data=(X_test, y_test))

# Eğitim doğruluğu ve doğrulama doğruluğu grafiği
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Eğitim kaybı ve doğrulama kaybı grafiği
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kaybı')
plt.legend()
plt.show()
# Modelin test setindeki tahminleri
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Precision, Recall ve F1-Skoru dahil değerlendirme raporu
print(classification_report(y_test, y_pred, target_names=["Olumsuz", "Olumlu"]))