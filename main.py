from ultralytics import YOLO
import cv2
import easyocr
import string

# Load YOLO model for license plate detection
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Path to the image file
image_path = 'D:\Program\MachineLearning\RecognitionLicensePlate\plat1.jpeg'

def read_license_plate(license_plate_crop):
    # Use EasyOCR to read text from license plate crop
    detections = reader.readtext(license_plate_crop)

    results = []
    for bbox, text, score in detections:
        # Clean up text and check if it complies with license plate format
        text = text.upper().replace(' ', '')
        if text != None:
            results.append((text, score))

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0], results[0][1]

    return None, None

# Read the image
frame = cv2.imread(image_path)
    
# Detect license plates in the frame
licensePlates = license_plate_detector(frame)[0]
    
# Process each detected license plate
for licensePlate in licensePlates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = licensePlate
            
    # Draw rectangle around license plate
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
    # Extract license plate region from the frame
    licensePlateCrop = frame[int(y1):int(y2), int(x1):int(x2), :]
        
    # Convert license plate region to grayscale
    licensePlateCropGray = cv2.cvtColor(licensePlateCrop, cv2.COLOR_BGR2GRAY)
        
    # Apply Otsu's thresholding to license plate region
    _, licensePlateCropThreshold = cv2.threshold(licensePlateCropGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Read license plate text using EasyOCR
    plate_text, plate_score = read_license_plate(licensePlateCropThreshold)
    print(plate_text, plate_score)
        
    # Display license plate text if found
    if plate_text:
        cv2.putText(frame, plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display thresholded license plate region
    cv2.imshow('License Plate Threshold', licensePlateCropThreshold)

# Display frame with detected license plates
cv2.imshow('Frame', frame)

# Wait for a key press and then close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()
