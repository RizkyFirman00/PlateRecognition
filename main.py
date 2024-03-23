import cv2
import easyocr
from ultralytics import YOLO  # Import YOLO at the time of use

def preprocess_image(image):
    """
    Sharpens, reduces noise, and normalizes an image.

    Args:
        image: The image to preprocess.

    Returns:
        The preprocessed image.
    """

    sharpened_image = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened_image = cv2.addWeighted(image, 1.5, sharpened_image, -0.5, 0)

    denoised_image = cv2.bilateralFilter(sharpened_image, 9, 75, 75)

    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quadrilateral = None
    for contour in contours:
        if len(contour) == 4:
            quadrilateral = contour
            break

    if quadrilateral is not None:
        warped_image = cv2.warpPerspective(denoised_image, cv2.getPerspectiveTransform(quadrilateral.reshape(4, 2), np.array([[0, 0], [200, 0], [200, 100], [0, 100]])), (200, 100))
    else:
        warped_image = denoised_image

    normalized_image = cv2.resize(warped_image, (200, 100))
    return normalized_image

def read_license_plate(image):
  """
  Reads license plate text from an image using EasyOCR.

  Args:
      image: The preprocessed image of the license plate.

  Returns:
      The extracted license plate text, or None if not found.
  """

  reader = easyocr.Reader(['id'], gpu=True)
  results = reader.readtext(image)

  plate_text = None
  for bbox, text, score in results:
      # Print bounding box coordinates and score for debugging
      print("Bounding Box:", bbox)
      print("Text:", text)
      print("Score:", score)

      formatted_text = text = text.upper().replace(' ', '')
      plate_text = formatted_text
      break

  return plate_text


def split_license_plate(plate_text):
    """
    Splits the license plate text into segments based on Indonesian format variations.

    Args:
        plate_text: The extracted license plate text.

    Returns:
        A tuple containing the kode wilayah (if any), nomor registrasi, and huruf akhir, or None if invalid format.
    """

    # Minimum and maximum allowed lengths for different parts
    min_kode_wilayah_length = 1
    max_kode_wilayah_length = 3
    min_nomor_registrasi_length = 1
    max_nomor_registrasi_length = 4
    min_huruf_akhir_length = 2
    max_huruf_akhir_length = 3

    kode_wilayah = ""
    nomor_registrasi = ""
    huruf_akhir = ""

    # Check for minimum plate length
    if len(plate_text) < min_kode_wilayah_length + min_nomor_registrasi_length + min_huruf_akhir_length:
        return None, None, None

    # Identify format based on first character
    if plate_text[0].isalpha():
        # Format: Kode Wilayah - Nomor Registrasi - Huruf Akhir
        kode_wilayah_length = min(len(plate_text), max_kode_wilayah_length)
        kode_wilayah = plate_text[:kode_wilayah_length]
        remaining_text = plate_text[kode_wilayah_length:]
    elif plate_text[1].isalpha():
        # Format: Nomor Registrasi - Kode Wilayah - Huruf Akhir
        kode_wilayah_length = min(len(plate_text) - 1, max_kode_wilayah_length)
        kode_wilayah = plate_text[1:kode_wilayah_length + 1]
        remaining_text = plate_text[kode_wilayah_length + 1:]
    else:
        # Format: Nomor Registrasi - Huruf Akhir
        remaining_text = plate_text

    # Extract nomor registrasi
    i = 0
    while i < len(remaining_text) and remaining_text[i].isdigit():
        nomor_registrasi += remaining_text[i]
        i += 1

    # Extract huruf akhir
    remaining_text = remaining_text[i:]
    while i < len(remaining_text) and remaining_text[i].isalpha():
        huruf_akhir += remaining_text[i]
        i += 1

    # Validate extracted segments
    if (len(kode_wilayah) not in range(min_kode_wilayah_length, max_kode_wilayah_length + 1) or
        len(nomor_registrasi) not in range(min_nomor_registrasi_length, max_nomor_registrasi_length + 1) or
        len(huruf_akhir) not in range(min_huruf_akhir_length, max_huruf_akhir_length + 1)):
        return None, None, None

    return kode_wilayah, nomor_registrasi, huruf_akhir


def detect_and_read_license_plates(image_path):
    """
    Detects license plates using YOLO, reads text using EasyOCR, and splits into segments.

    Args:
        image_path: The path to the image file.

    Returns:
        None
    """

    # Load YOLO model (assuming it's in the same directory)
    license_plate_detector = YOLO('./models/license_plate_detector.pt')

    # Read the image
    frame = cv2.imread(image_path)

    # Detect license plates
    licensePlates = license_plate_detector(frame)[0]

    # Process each detected license plate
    for licensePlate in licensePlates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = licensePlate

        # Draw rectangle around license plate
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Extract license plate region
        licensePlateCrop = frame[int(y1):int(y2), int(x1):int(x2), :]

        # Preprocess the license plate region
        preprocessed_plate = preprocess_image(licensePlateCrop)

        # Read license plate text
        plate_text = read_license_plate(preprocessed_plate)

        # Split license plate text (if valid)
        if plate_text:
            kode_wilayah, nomor_registrasi, huruf_akhir = split_license_plate(plate_text)

            print(f"Plat Nomor: {plate_text}")
            print(f"Kode Wilayah: {kode_wilayah}")
            print(f"Nomor Registrasi: {nomor_registrasi}")
            print(f"Huruf Akhir: {huruf_akhir}")

            # Display license plate text on the frame
            cv2.putText(frame, plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display thresholded license plate region for debugging (optional)
        cv2.imshow('License Plate Threshold', preprocessed_plate)

    # Display frame with detected license plates
    cv2.imshow('Frame', frame)

    # Wait for a key press and then close all OpenCV windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'D:\Program\MachineLearning\RecognitionLicensePlate\plat4.jpg'
detect_and_read_license_plates(image_path)