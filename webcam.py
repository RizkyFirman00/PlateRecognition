import cv2

# Fungsi untuk menampilkan video dari webcam
def show_webcam():
    # Inisialisasi objek VideoCapture untuk mengakses feed dari webcam (nomor kamera default adalah 0)
    cap = cv2.VideoCapture(0)

    # Loop tak terbatas untuk menampilkan frame dari feed webcam
    while True:
        # Membaca frame dari feed webcam
        ret, frame = cap.read()

        # Menampilkan frame yang dibaca
        cv2.imshow('Webcam', frame)

        # Tunggu tombol kunci 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Setelah keluar dari loop, stop feed webcam dan tutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()

# Memanggil fungsi untuk menampilkan video dari webcam
show_webcam()
