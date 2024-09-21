from iteung import reply
import pandas as pd
import os

#Kode ini mengecek apakah file Excel (hasil_akurasi_bot.xlsx) sudah ada atau tidak. 
#Jika ada, maka data dari file tersebut akan dibaca dan dimuat ke dalam beberapa list 
#(seperti listJawaban, listAkurasi, dan listPertanyaan). Jika file kosong, akan ditangkap dengan penanganan kesalahan pd.errors.EmptyDataError.
file_path = "hasil_dataset_bot.xlsx"
adaFile = False

listJawaban = []
listAkurasi = []
listPertanyaan = []

if os.path.exists(file_path):
    print("File Excel ada.")
    try:
        # Membaca data dari file Excel
        data = pd.read_excel(file_path)
        # Mengambil data dari kolom yang ditentukan
        listJawaban = data["Jawaban"].tolist()
        listAkurasi = data["Akurasi"].tolist()
        listPertanyaan = data["Pertanyaan"].tolist()
        adaFile = True
    except pd.errors.EmptyDataError:
        print("File Excel Kosong.")
else:
    print("File Excel tidak ada.")

#Bagian ini berisi loop tak terbatas yang memungkinkan pengguna berinteraksi dengan bot. 
#Setiap kali pengguna memasukkan pesan, bot memberikan jawaban melalui fungsi reply.botReply. 
#Jawaban tersebut, bersama dengan pertanyaan dan akurasi, dicatat dalam list terkait. 
#Selanjutnya, data tersebut disimpan kembali ke dalam file Excel (hasil_akurasi_bot.xlsx). 
#Jika file sudah ada, akan dilakukan penulisan tanpa indeks. 
#Jika file belum ada, akan dibuat file baru dengan indeks yang diabaikan.
while True:
    message = input("Kamu: ")
    #Jika pengguna memasukkan pesan "exit", loop akan dihentikan, dan program keluar.
    if message == "exit":

        break
    return_message, status , dec_outputs, akurasi= reply.botReply(message)
    listJawaban.append(return_message)
    listAkurasi.append(akurasi)
    listPertanyaan.append(message)

    print(f"ITeung: {return_message}")

    df = pd.DataFrame({
        'Pertanyaan': listPertanyaan,
        'Jawaban': listJawaban,
        'Akurasi': listAkurasi
    })

    if adaFile:
        try:
            df.to_excel(file_path, index=False)
        except PermissionError:
            print("File Excel sedang dibuka. Tutup file tersebut dan coba lagi.")

    else:
        df.to_excel(file_path, index=False)
