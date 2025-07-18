
# Prompt to GPT4: write a python script that loads a .mov file and extracts one RGB image in cv2
# image format every second. Next, pytesseract is used to extract bounding boxes of the text displayed
# in the image. The code then loops over all bounding boxes and extracts text in Japanese language from them.
# bounding box positions and text are printed to the console

import cv2
import torch
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import deque
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

#Disable parallelism for CPU mode
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def calculate_total_bounding_box_percentile(d, draw, image_width, image_height, debug = False):
    x_coords = []
    y_coords = []
    w_coords = []
    h_coords = []
    limx = 800
    limy = 80

    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        text = d['text'][i]

        # Filter out empty text
        if text.strip() and (h < limy) and (w < limx):
            x_coords.append(x)
            y_coords.append(y)
            w_coords.append(w)
            h_coords.append(h)
            if debug:
                draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
        if text.strip() and ((h > limy) or (w > limx)):
            if debug:
                draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)

    if not x_coords:
        return None, None, None, None

    # Sort the coordinates
    x_coords.sort()
    y_coords.sort()
    w_coords.sort()
    h_coords.sort()

    # Calculate percentiles
    total_x1 = np.percentile(x_coords, 10)
    total_y1 = np.percentile(y_coords, 10)
    total_x2 = np.percentile([x + w for x, w in zip(x_coords, w_coords)], 90)
    total_y2 = np.percentile([y + h for y, h in zip(y_coords, h_coords)], 90)

    # Add margins
    margin_x = 0.05 * image_width
    margin_y = 0.05 * image_height

    total_x1 = max(0, total_x1)
    total_y1 = max(0, total_y1)
    total_x2 = min(image_width, total_x2 + margin_x)
    total_y2 = min(image_height, total_y2 + margin_y)

    return total_x1, total_y1, total_x2, total_y2





def average_bounding_boxes(history):
    valid_entries = [box for box in history if box[0] is not None]
    if len(valid_entries) == 0 or len(valid_entries) < len(history) // 2:
        return None, None, None, None

    avg_x1 = np.median([box[0] for box in valid_entries])
    avg_y1 = np.median([box[1] for box in valid_entries])
    avg_x2 = np.median([box[2] for box in valid_entries])
    avg_y2 = np.median([box[3] for box in valid_entries])

    return avg_x1, avg_y1, avg_x2, avg_y2


def add_newline_after_center_word(text):
    words = text.split()
    if len(words) <= 1:
        return text  # No center word if there's one or zero words

    center_index = len(words) // 2
    # Rebuild the text with a newline after the center word
    new_text = ' '.join(words[:center_index + 1]) + '\n' + ' '.join(words[center_index + 1:])
    return new_text

# Initialize video capture
video_path = 'JapanRPG_TestSequence.mov'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps/4)  # Interval in frames to extract one image per second

# Tesseract configuration for Japanese language
tess_config_initial = '--psm 3 -l jpn'  # Initial configuration
tess_config_refined = '--psm 6 -l jpn'  # Refined configuration for block of text


# Load a font that supports Japanese characters
font_path = 'NotoMono-Regular.ttf'  # Replace with the path to a suitable font file
font = ImageFont.truetype(font_path, 20)

min_box_width = 0
frame_count = 0
image_height, image_width = 640, 480

tokenizer = AutoTokenizer.from_pretrained("nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("nllb-200-distilled-600M")

translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='jpn_Jpan', tgt_lang='eng_Latn',
                      max_length=200)

history = deque(maxlen=10)  # To store the history of bounding boxes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Extract image every second
        image = frame
        if frame_count == 0:
            image_height, image_width = image.shape[:2]
            min_box_width = 0.2 * image_width

        # Use pytesseract to get bounding boxes of text
        d = pytesseract.image_to_data(image, config=tess_config_initial, output_type=Output.DICT)

        # Convert OpenCV image to PIL image
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        n_boxes = len(d['level'])
        # Calculate the total bounding box
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        total_x1, total_y1, total_x2, total_y2 = calculate_total_bounding_box_percentile(d, draw, image_width, image_height)

        # Append the current bounding box to the history
        history.append((total_x1, total_y1, total_x2, total_y2))

        # Calculate the average bounding box from the history
        avg_x1, avg_y1, avg_x2, avg_y2 = average_bounding_boxes(history)

        # Crop the region of interest (ROI) for the detected text block
        if total_x1 is not None and total_x2 is not None and total_y1 is not None and total_y2 is not None:
            if total_x1 < total_x2 and total_y1 < total_y2 and (total_x2 - total_x1 > min_box_width):
                roi = image[int(total_y1):int(total_y2), int(total_x1):int(total_x2)]

                # Second pass: Refined OCR on the ROI
                refined_text = pytesseract.image_to_string(roi, config=tess_config_refined)

                # Compute the average color in the bounding box
                avg_color = cv2.mean(image[int(total_y1):int(total_y2), int(total_x1):int(total_x2)])[:3]
                avg_color = tuple(map(int, avg_color))

                # Set the entire bounding box to the average color
                image[int(total_y1):int(total_y2), int(total_x1):int(total_x2)] = avg_color

                #Convert to PIL Image
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image_pil)

                # Display the refined text
                print(f"Refined Text: {refined_text}")

                # Translate and print
                translation = translator(refined_text)[0]
                trans_text = add_newline_after_center_word(translation['translation_text'])
                print(f"Translated Text: {trans_text}")

                # Calculate the bounding box size
                bbox_width = total_x2 - total_x1
                bbox_height = total_y2 - total_y1

                # Find the appropriate font size that fits the bounding box
                font_size = 20
                font = ImageFont.truetype(font_path, font_size)
                error = False
                while True:
                    try:
                        text_bbox = font.getbbox(trans_text)
                    except ValueError:
                        error = True
                        break
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    if text_width < bbox_width and text_height < bbox_height:
                        font_size += 1
                        font = ImageFont.truetype(font_path, font_size)
                    else:
                        font_size -= 1
                        font = ImageFont.truetype(font_path, font_size)
                        break

                # Draw the translated text
                if not error:
                    draw.text((total_x1, total_y1), trans_text, font=font, fill=(255, 255, 255))

                # Convert back to OpenCV format
                image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Convert PIL image back to OpenCV format
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


        # Display the image with bounding boxes
        cv2.imshow('Frame', image)
        cv2.waitKey(int(frame_interval / fps * 10))


    frame_count += 1

    if frame_count > 2000:
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
