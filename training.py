# -*- coding: utf-8 -*-
#Pada bagian ini, pustaka-pustaka dan modul-modul yang diperlukan diimpor, termasuk TensorFlow dan Keras untuk membangun dan melatih model.
import json
import os
import pickle

import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.activations import softmax
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Concatenate
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

#Ini menginisialisasi sesi TensorFlow untuk log perangkat yang digunakan.
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#Membuat direktori keluaran untuk menyimpan file konfigurasi, log, dan model.
path = "output_dir2/"
try:
    os.makedirs(path)
except:
    pass

#Membaca file teks yang berisi pasangan pertanyaan dan jawaban yang sudah dibersihkan.
dataset = pd.read_csv('./dataset/clean_qa.txt', delimiter="|", header=None,lineterminator='\n')

#Membagi data menjadi set pelatihan dan pengujian.
dataset_val = dataset.iloc[412:].to_csv('output_dir2/val.csv')

dataset_train = dataset.iloc[:412]

questions_train = dataset_train.iloc[:, 0].values.tolist()
answers_train = dataset_train.iloc[:, 1].values.tolist()

questions_test = dataset_train.iloc[:, 0].values.tolist()
answers_test = dataset_train.iloc[:, 1].values.tolist()


def save_tokenizer(tokenizer):
    with open('output_dir2/tokenizer.pickle', 'wb') as handle: # Membuka file 'output_dir2/tokenizer.pickle' dalam mode penulisan biner ('wb'). Penanda as handle digunakan untuk menunjukkan file yang sedang dibuka.
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) #Menggunakan modul pickle untuk menyimpan objek tokenizer ke dalam file yang dibuka. Penggunaan pickle.HIGHEST_PROTOCOL menentukan protokol tertinggi untuk serialisasi objek.


def save_config(key, value):
    data = {} #Membuat kamus kosong dengan nama data, yang akan digunakan untuk menyimpan konfigurasi.
    if os.path.exists(path + 'config.json'): #Memeriksa apakah file 'config.json' sudah ada dalam direktori yang ditentukan oleh variabel path. os.path.exists() mengembalikan True jika file ada dan False jika tidak.
        with open(path + 'config.json') as json_file: #Membuka file 'config.json' untuk dibaca. Penanda as json_file digunakan untuk menunjukkan file yang sedang dibuka.
            data = json.load(json_file) #Membuka file 'config.json' untuk dibaca. Penanda as json_file digunakan untuk menunjukkan file yang sedang dibuka.

    data[key] = value #Menetapkan nilai value ke kunci key dalam kamus data.
    with open(path + 'config.json', 'w') as outfile: #Membuka file 'config.json' dalam mode penulisan. Penanda as outfile digunakan untuk menunjukkan file yang sedang dibuka.
        json.dump(data, outfile) #Menggunakan modul json untuk menulis isi kamus data ke dalam file JSON yang dibuka.



target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789' #dihapus atau diganti dalam proses pembersihan teks

#Menggunakan Tokenizer untuk mengonversi teks menjadi urutan token dan melakukan padding agar 
#semua urutan memiliki panjang yang sama.
tokenizer = Tokenizer(filters=target_regex, lower=True) #Membuat objek tokenizer dengan mengatur filter karakter yang akan dihapus sesuai dengan target_regex, dan mengonversi semua teks menjadi huruf kecil selama tokenisasi.
tokenizer.fit_on_texts(questions_train + answers_train + questions_test + answers_test) #Melatih tokenizer pada teks yang diberikan, yaitu gabungan dari pertanyaan dan jawaban dari set data pelatihan (questions_train dan answers_train) serta set data pengujian (questions_test dan answers_test).
save_tokenizer(tokenizer) #Menyimpan objek tokenizer ke dalam file sistem file menggunakan fungsi save_tokenizer().

VOCAB_SIZE = len(tokenizer.word_index) + 1 #Menghitung ukuran vocab (kosa kata) yang dihasilkan dari tokenizer dengan menambahkan 1 untuk mengakomodasi indeks nol sebagai placeholder untuk padding.
save_config('VOCAB_SIZE', VOCAB_SIZE) #Menyimpan konfigurasi VOCAB_SIZE ke dalam file konfigurasi menggunakan fungsi save_config().
print('Vocabulary size : {}'.format(VOCAB_SIZE)) #Mencetak ukuran vocab yang dihitung ke konsol untuk memberikan informasi kepada pengguna.

tokenized_questions_train = tokenizer.texts_to_sequences(questions_train) #Mengonversi teks pertanyaan dalam set data pelatihan (questions_train) menjadi urutan bilangan bulat sesuai dengan indeks yang diberikan oleh tokenizer.
maxlen_questions_train = max([len(x) for x in tokenized_questions_train]) #Menghitung panjang maksimum dari pertanyaan yang telah diurutkan dalam bentuk token bilangan bulat.
save_config('maxlen_questions', maxlen_questions_train) #Menyimpan panjang maksimum dari pertanyaan ke dalam file konfigurasi menggunakan fungsi save_config().
encoder_input_data_train = pad_sequences(tokenized_questions_train, maxlen=maxlen_questions_train, padding='post') # Mengisi atau memotong urutan token bilangan bulat dari pertanyaan dalam set data pelatihan menjadi panjang maksimum yang telah ditentukan (maxlen_questions_train). 

tokenized_questions_test = tokenizer.texts_to_sequences(questions_test)
maxlen_questions_test = max([len(x) for x in tokenized_questions_test])
save_config('maxlen_questions', maxlen_questions_test)
encoder_input_data_test = pad_sequences(tokenized_questions_test, maxlen=maxlen_questions_test, padding='post')

tokenized_answers_train = tokenizer.texts_to_sequences(answers_train)
maxlen_answers_train = max([len(x) for x in tokenized_answers_train])
save_config('maxlen_answers', maxlen_answers_train)
decoder_input_data_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')

tokenized_answers_test = tokenizer.texts_to_sequences(answers_test)
maxlen_answers_test = max([len(x) for x in tokenized_answers_test])
save_config('maxlen_answers', maxlen_answers_test)
decoder_input_data_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')

for i in range(len(tokenized_answers_train)):
    tokenized_answers_train[i] = tokenized_answers_train[i][1:] #Memotong setiap urutan token jawaban dalam tokenized_answers_train untuk menghapus token pertama. 
    #Ini dilakukan karena kita ingin menggunakan urutan jawaban sebagai input untuk decoder dan mengabaikan token pertama yang merupakan token START.
padded_answers_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post') 
#Mengisi atau memotong urutan token jawaban yang telah dipotong dalam tokenized_answers_train sehingga memiliki panjang maksimum yang telah ditentukan 
decoder_output_data_train = to_categorical(padded_answers_train, num_classes=VOCAB_SIZE) #Mengonversi urutan token jawaban yang telah di-padding ke dalam bentuk one-hot encoding dengan menggunakan fungsi to_categorical

for i in range(len(tokenized_answers_test)):
    tokenized_answers_test[i] = tokenized_answers_test[i][1:]
padded_answers_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')
decoder_output_data_test = to_categorical(padded_answers_test, num_classes=VOCAB_SIZE)

#Membangun bagian encoder dari model seq2seq dengan menggunakan layer-layer LSTM dan layer-layer Embedding.
enc_inp = Input(shape=(None,))
enc_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(enc_inp)
enc_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(256, return_state=True, dropout=0.5, recurrent_dropout=0.5))(enc_embedding)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
enc_states = [state_h, state_c]

#Membangun bagian decoder dari model seq2seq dengan menggunakan layer-layer LSTM dan layer-layer Embedding.
dec_inp = Input(shape=(None,))
dec_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(dec_inp)
dec_lstm = LSTM(256 * 2, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)
dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)

#Membangun layer Dense untuk output model dengan fungsi aktivasi softmax.
dec_dense = Dense(VOCAB_SIZE, activation=softmax)
output = dec_dense(dec_outputs)

#Membuat log TensorBoard dan checkpoint model untuk melacak dan menyimpan perkembangan pelatihan.
logdir = os.path.join(path, "logs")
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

checkpoint = ModelCheckpoint(os.path.join(path, 'model-{epoch:02d}-{loss:.2f}.hdf5'),
                             monitor='loss',
                             verbose=1,
                             save_best_only=True, mode='auto', period=100)

#Mengompilasi dan melatih model dengan menggunakan data pelatihan dan validasi.
model = Model([enc_inp, dec_inp], output)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

batch_size = 20
epochs = 800
model.fit([encoder_input_data_train, decoder_input_data_train],
          decoder_output_data_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([encoder_input_data_test, decoder_input_data_test], decoder_output_data_test),
          callbacks=[tensorboard_callback, checkpoint])
#Menyimpan model setelah pelatihan selesai.
model.save(os.path.join(path, 'model-' + path.replace("/", "") + '.h5'))