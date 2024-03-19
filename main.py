from ultralytics import YOLO
import cv2
import easyocr
import string
import os

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

# Load YOLO model for license plate detection
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Open video capture (assuming it's from webcam)
cap = cv2.VideoCapture(0)

# Create directory to save images if it doesn't exist
save_dir = 'detected_plates'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def read_license_plate(license_plate_crop):
    # Use EasyOCR to read text from license plate crop
    detections = reader.readtext(license_plate_crop)

    for bbox, text, score in detections:
        # Clean up text and check if it complies with license plate format
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score

    return None, None

def format_license(text):
    # Format the license plate text
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char, 7: dict_int_to_char,
               1: dict_char_to_int, 2: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6, 7]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def license_complies_format(text):
    # Check if the license plate text complies with the required format
    if len(text) != 8:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()) and \
       (text[7] in string.ascii_uppercase or text[7] in dict_int_to_char.keys()):
        return True
    else:
        return False

# Main loop to process video frames
while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)  # 1 for horizontal flip, 0 for vertical flip, -1 for both horizontal and vertical flip
    
    # Detect license plates in the frame
    licensePlates = license_plate_detector(frame)[0]
    for licensePlate in licensePlates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = licensePlate
            
        # Draw rectangle around license plate
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Extract license plate region from the frame
        licensePlateCrop = frame[int(y1):int(y2), int(x1):int(x2), :]
        
        # Convert license plate region to grayscale
        licensePlateCropGray = cv2.cvtColor(licensePlateCrop, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to license plate region
        _, licensePlateCropThreshold = cv2.threshold(licensePlateCropGray, 100, 255, cv2.THRESH_BINARY_INV)

        # Read license plate text using EasyOCR
        plate_text, plate_score = read_license_plate(licensePlateCropThreshold)
        
        # Display license plate text if found
        if plate_text:
            # Save the frame containing the license plate
            filename = os.path.join(save_dir, f'plate_{plate_text}.jpg')
            cv2.imwrite(filename, licensePlateCrop)
            print(f"License Plate Text: {plate_text}, Saved as: {filename}")
            
            # Save the license plate text to a text file
            with open('detected_plates.txt', 'a') as f:
                f.write(f'License Plate: {plate_text}, Score: {plate_score}\n')

            # Draw the detected license plate text on the frame
            cv2.putText(frame, plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display license plate region in grayscale
        cv2.imshow('License Plate Gray', licensePlateCropGray)
        
        # Display thresholded license plate region
        cv2.imshow('License Plate Threshold', licensePlateCropThreshold)

    # Display frame with detected license plates
    cv2.imshow('Frame', frame)

        
       
