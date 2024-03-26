import cv2
import easyocr
from ultralytics import YOLO  # Import YOLO at the time of use
import numpy as np
import re

def preprocess_image(image):
    sharpened_image = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened_image = cv2.addWeighted(image, 1.5, sharpened_image, -0.5, 0)

    denoised_image = cv2.bilateralFilter(sharpened_image, 9, 75, 75)

    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quadrilateral = None
    for contour in contours:
        if len(contour) == 4:
            quadrilateral = contour.reshape(4, 2)
            break

    if quadrilateral is not None:
        src_points = np.float32(quadrilateral)
        dst_points = np.float32([[0, 0], [200, 0], [200, 100], [0, 100]])
        warped_image = cv2.warpPerspective(denoised_image, cv2.getPerspectiveTransform(src_points, dst_points), (200, 100))
    else:
        warped_image = denoised_image

    normalized_image = cv2.resize(warped_image, (200, 100))
    return normalized_image

def read_license_plate(image):
    reader = easyocr.Reader(['id'], gpu=True)
    results = reader.readtext(image)

    plate_text = None
    for bbox, text, score in results:
        # Remove non-alphanumeric characters
        cleaned_text = re.sub('[\W_]+', '', text)
        print("Bounding Box:", bbox)
        print("Text:", cleaned_text)
        print("Score:", score)

        formatted_text = cleaned_text.upper()
        plate_text = formatted_text
        break

    return plate_text


def split_license_plate(plate_text):
    min_kode_wilayah_length = 1
    max_kode_wilayah_length = 3
    min_nomor_registrasi_length = 1
    max_nomor_registrasi_length = 4
    min_huruf_akhir_length = 2
    max_huruf_akhir_length = 3

    kode_wilayah = ""
    nomor_registrasi = ""
    huruf_akhir = ""

    if len(plate_text) < min_kode_wilayah_length + min_nomor_registrasi_length + min_huruf_akhir_length:
        return None, None, None

    if plate_text[0].isalpha():
        kode_wilayah_length = min(len(plate_text), max_kode_wilayah_length)
        kode_wilayah = plate_text[:kode_wilayah_length]
        remaining_text = plate_text[kode_wilayah_length:]
    elif plate_text[1].isalpha():
        kode_wilayah_length = min(len(plate_text) - 1, max_kode_wilayah_length)
        kode_wilayah = plate_text[1:kode_wilayah_length + 1]
        remaining_text = plate_text[kode_wilayah_length + 1:]
    else:
        remaining_text = plate_text

    i = 0
    while i < len(remaining_text) and remaining_text[i].isdigit():
        nomor_registrasi += remaining_text[i]
        i += 1

    remaining_text = remaining_text[i:]
    while i < len(remaining_text) and remaining_text[i].isalpha():
        huruf_akhir += remaining_text[i]
        i += 1

    # Validate extracted segments
    if (len(kode_wilayah) < min_kode_wilayah_length or len(kode_wilayah) > max_kode_wilayah_length or
        len(nomor_registrasi) < min_nomor_registrasi_length or len(nomor_registrasi) > max_nomor_registrasi_length or
        len(huruf_akhir) < min_huruf_akhir_length or len(huruf_akhir) > max_huruf_akhir_length):
        return None, None, None

    return kode_wilayah, nomor_registrasi, huruf_akhir

def detect_and_read_license_plates(image_path):
    license_plate_detector = YOLO('./models/license_plate_detector.pt')

    frame = cv2.imread(image_path)
    licensePlates = license_plate_detector(frame)[0]

    for licensePlate in licensePlates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = licensePlate

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        licensePlateCrop = frame[int(y1):int(y2), int(x1):int(x2), :]

        preprocessed_plate = preprocess_image(licensePlateCrop)

        plate_text = read_license_plate(preprocessed_plate)

        if plate_text:
            kode_wilayah, nomor_registrasi, huruf_akhir = split_license_plate(plate_text)

            print(f"Plat Nomor: {plate_text}")
            print(f"Kode Wilayah: {kode_wilayah}")
            print(f"Nomor Registrasi: {nomor_registrasi}")
            print(f"Huruf Akhir: {huruf_akhir}")

            text_to_display = f"{plate_text} ({score*100:.2f}%)"
            cv2.putText(frame, text_to_display, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('License Plate Threshold', preprocessed_plate)

    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'D:\Program\MachineLearning\RecognitionLicensePlate\plat.jpeg'
detect_and_read_license_plates(image_path)
