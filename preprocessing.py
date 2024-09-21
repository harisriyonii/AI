# -*- coding: utf-8 -*-
"""ITeung

# Preprocessing
"""

#Baris ini mengimpor pustaka dan modul yang diperlukan untuk pra-pemrosesan teks dan manipulasi 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #sebuah kelas untuk membuat objek stemmer untuk bahasa indonesia

import io #untuk membaca data dari file tekx, koneksi jaringan dan buffer dalam memori
import os #Mengakses jalur file, memeriksa keberadaan file, membuat/menghapus file dan direktori
import re #regular expressions untuk membantu mencari pola tertentu di dalam string misalnya memvalidasi input, mengekstrak data dari teks
import requests #membantu permintaan HTTP ke server web 
import csv #untuk membaca data dari file csv dan menulis data dari struktur data python ke file csv
import datetime #untuk mendapatkan tanggal dan waktu saat ini
import numpy as np #untuk operasi matematika, membuat dan mamanipulasi array multidemensi
import pandas as pd # membuat dan memanipulasi data 
import random #memilih sampel acak dari kumpulan data
import pickle #menyimpan objek python dan berbagi objek python anatar proses atau di seluruh mesin

#Kode ini menyiapkan stemmer Sastrawi, alat untuk melakukan stemming pada kata-kata dalam bahasa Indonesia.
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#Baris ini membuat pola ekspresi reguler untuk menghapus tanda baca dari kalimat dan menentukan daftar string yang mewakili frasa tidak dikenal.
punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
unknowns = ["gak paham","kurang ngerti","I don't know"]


#Membaca csv dan mengubahnya menjadi array NumPy.
list_indonesia_slang = pd.read_csv('./dataset/daftar-slang-bahasa-indonesia.csv', header=None).to_numpy()

#Setelah kode ini dijalankan, kamus data_slang akan berisi semua kata slang dan maknanya dari file CSV.
data_slang = {}
for key, value in list_indonesia_slang:
    data_slang[key] = value

#Fungsi-fungsi ini digunakan untuk menangani kata-kata slang dan normalisasi kalimat.
def dynamic_switcher(dict_data, key):
    return dict_data.get(key, None)

def check_normal_word(word_input):
    slang_result = dynamic_switcher(data_slang, word_input)
    if slang_result:
        return slang_result
    return word_input

#Fungsi ini melakukan normalisasi kalimat dengan menghapus kata-kata tertentu, mengganti slang, dan melakukan stemming.
def normalize_sentence(sentence):
  sentence = punct_re_escape.sub('', sentence.lower())
  sentence = sentence.replace('iteung', '').replace('\n', '').replace(' wah','').replace('wow','').replace(' dong','').replace(' sih','').replace(' deh','')
  sentence = sentence.replace('teung', '')
  sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence)
  sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence)
  sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence)
  sentence = ' '.join(sentence.split())
  if sentence:
    sentence = sentence.strip().split(" ")
    normal_sentence = " "
    for word in sentence:
      normalize_word = check_normal_word(word)
      root_sentence = stemmer.stem(normalize_word)
      normal_sentence += root_sentence+" "
    return punct_re_escape.sub('',normal_sentence)
  return sentence

#Membaca file CSV yang berisi pasangan pertanyaan dan jawaban.
df = pd.read_csv('./dataset/qa.csv', sep='|',usecols= ['question','answer'])
df.head()

#Menghitung panjang pertanyaan dan jawaban, menyimpan hasilnya dalam bentuk kamus.
question_length = {} #membuat dua kamus kosong yang akan digunakan untuk menyimpan pertanyaan dan jawaban
answer_length = {}

for index, row in df.iterrows(): #melakukan iterasi melalui setiap baris dalam DataFrame df
  question = normalize_sentence(row['question']) #dijalankan dua kali. melakukan pembersihan teks, seperti menghilangkan tanda baca atau huruf kapital
  question = normalize_sentence(question) #untuk memperoleh pernyaan yang dinormalisasi
  question = stemmer.stem(question) #Mengaplikasikan stemming pada pertanyaan yang dinormalisasi menggunakan objek stemmer.

  if question_length.get(len(question.split())): #Memeriksa apakah panjang pertanyaan (dalam jumlah kata) sudah ada dalam kamus question_length.
    question_length[len(question.split())] += 1 #Jika panjang pertanyaan sudah ada dalam kamus, tambahkan satu ke nilai yang ada. Jika belum, tambahkan panjang pertanyaan sebagai kunci baru dengan nilai 1.
  else:
    question_length[len(question.split())] = 1 #Sama seperti langkah 6, tetapi untuk panjang jawaban.

  if answer_length.get(len(str(row['answer']).split())):
    answer_length[len(str(row['answer']).split())] += 1 #Sama seperti langkah 7, tetapi untuk jawaban.
  else:
    answer_length[len(str(row['answer']).split())] = 1

question_length #Mengembalikan kamus yang berisi panjang pertanyaan (dalam jumlah kata) dan frekuensinya.

answer_length #Mengembalikan kamus yang berisi panjang jawaban (dalam jumlah kata) dan frekuensinya.

# Membuat list dari nilai panjang pertanyaan, kunci panjang pertanyaan, dan pasangan kunci-nilai panjang pertanyaan.
val_question_length = list(question_length.values())
key_question_length = list(question_length.keys())
key_val_question_length = list(zip(key_question_length, val_question_length))
#Membuat DataFrame dari pasangan kunci-nilai panjang pertanyaan, dengan kolom 'length_data' untuk panjang dan 'total_sentences' untuk frekuensi.
df_question_length = pd.DataFrame(key_val_question_length, columns=['length_data', 'total_sentences']) 
df_question_length.sort_values(by=['length_data'], inplace=True) #Mengurutkan DataFrame berdasarkan panjang pertanyaan.
df_question_length.describe() #Memberikan ringkasan statistik tentang panjang pertanyaan dalam DataFrame.

#Baris ini membuat list baru val_question_length yang berisi semua nilai (jumlah kemunculan) dari dictionary question_length.
val_answer_length = list(answer_length.values())
#Baris ini membuat list baru key_question_length yang berisi semua kunci (panjang kalimat) dari dictionary question_length
key_answer_length = list(answer_length.keys()) 
key_val_answer_length = list(zip(key_answer_length, val_answer_length)) #Baris ini menggunakan fungsi zip untuk menggabungkan list key_question_length dan val_question_length menjadi list of tuples (daftar tupel) key_val_question_length. Setiap tupel berisi pasangan (kunci, nilai).
df_answer_length = pd.DataFrame(key_val_answer_length, columns=['length_data', 'total_sentences']) #Baris ini membuat DataFrame baru bernama df_answer_length dari list of tuples key_val_answer_length. Kolom pertama diberi nama 'length_data' dan berisi panjang kalimat, dan kolom kedua 'total_sentences' berisi jumlah kemunculan setiap panjang kalimat.
df_answer_length.sort_values(by=['length_data'], inplace=True) #Baris ini mengurutkan DataFrame df_answer_length berdasarkan kolom 'length_data' (panjang kalimat).
df_answer_length.describe() #Baris ini menampilkan statistik deskriptif untuk DataFrame df_answer_length.

data_length = 0 #Membuat variabel data_length dengan nilai 0. 

#filename = open('./dataset/clean_qa.txt', 'a+')
#Normalisasi dan membersihkan pasangan pertanyaan-jawaban, menyaringnya berdasarkan kriteria panjang, dan menulis data yang telah dibersihkan ke file teks.
filename= './dataset/clean_qa.txt' #Baris ini mendefinisikan nama file tempat data yang diproses akan disimpan. Nama filenya adalah clean_qa.txt dan disimpan di folder ./dataset/.
with open(filename, 'w', encoding='utf-8') as f: # Membuka file filename dalam mode penulisan ('w') dengan encoding UTF-8. Ini memastikan bahwa data yang ditulis ke file akan dienkripsi menggunakan UTF-8. as f membuat variabel f yang merupakan penanda untuk file yang sedang dibuka.
  for index, row in df.iterrows(): #Melakukan iterasi melalui setiap baris dalam DataFrame df.
    question = normalize_sentence(str(row['question'])) # Mengambil nilai dari kolom 'question' pada baris saat ini, mengonversinya menjadi string, dan menerapkan fungsi normalize_sentence() untuk memperoleh pertanyaan yang dinormalisasi.
    question = normalize_sentence(question) #Ini tampaknya merupakan duplikasi dari baris sebelumnya. Mungkin perlu diperbaiki.
    question = stemmer.stem(question) #Mengaplikasikan stemming pada pertanyaan yang dinormalisasi menggunakan objek stemmer.

    answer = str(row['answer']).lower().replace('iteung', 'aku').replace('\n', ' ') #Mengambil nilai dari kolom 'answer' pada baris saat ini, mengonversinya menjadi string, mengonversi semuanya menjadi huruf kecil, dan mengganti 'iteung' dengan 'aku' (mungkin sebuah substitusi spesifik dalam konteks tertentu). Baris ini juga menghapus karakter newline ('\n') dan menggantinya dengan spasi.

    if len(question.split()) > 0 and len(question.split()) < 13 and len(answer.split()) < 29: #Memeriksa apakah jumlah kata dalam pertanyaan antara 1 dan 12 (inklusif) dan jumlah kata dalam jawaban kurang dari 29.
      body="{"+question+"}|<START> {"+answer+"} <END>"
     #Mencetak setiap pasangan pertanyaan-jawaban yang telah diproses ke file teks dalam format yang ditentukan.
      print(body, file=f)
      #filename.write(f"{question}\t<START> {answer} <END>\n")

#filename.close()
