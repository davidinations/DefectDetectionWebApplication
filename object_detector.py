from flask import Flask, request, Response, render_template, jsonify
# from flask_mysqldb import MySQL
from waitress import serve
from PIL import Image, ImageDraw
import json
import os
import webbrowser
from ultralytics import YOLO
import csv
from datetime import datetime

app = Flask(__name__)

# Get the absolute path to the directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the model path
model_path = os.path.join(script_dir, "StitchDetection.pt")
model = YOLO(model_path)

uploads_dir = os.path.join(script_dir, "Assets", "Uploads")
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

results_dir = os.path.join(script_dir, "Assets", "Results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

dataset_csv_path = os.path.join(script_dir, "Assets", "Dataset.csv")
if not os.path.exists(dataset_csv_path):
    os.makedirs(dataset_csv_path)


def get_next_filename():
    existing_files = os.listdir(uploads_dir)
    existing_numbers = [int(f.split('.')[0])
                        for f in existing_files if f.split('.')[0].isdigit()]
    next_number = max(existing_numbers, default=0) + 1
    return f"{next_number}.jpg"


@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    return render_template("index.html")


# @app.route('/history', methods=['GET'])
# def get_history():
#     history = []
#     with open(dataset_csv_path, mode='r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             history.append(row)
#     return jsonify(history)


@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file",
    saves it with an incrementing name, passes it through YOLOv8 object detection
    network and returns an array of bounding boxes.
    :return: a JSON array of objects bounding
    boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    filename = get_next_filename()
    file_path = os.path.join(uploads_dir, filename)
    buf.save(file_path)

    image = Image.open(file_path)
    boxes, count = detect_objects_on_image(image)
    response_data = {
        "count": count,
        "boxes": boxes
    }

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2, object_type, probability, xAxis, yAxis = box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        # draw.text((x1, y1), f"{object_type} {probability:.2f}", fill="green")

    # Save the result image in the Assets/Results directory
    result_file_path = os.path.join(
        results_dir, f"{filename.split('.')[0]}.jpg")
    image.save(result_file_path)

    # Append result to Dataset.csv
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(dataset_csv_path, mode='a', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow([file_path, result_file_path, count, current_time])

    return Response(
        json.dumps(response_data),
        mimetype='application/json'
    )


def detect_objects_on_image(image):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param image: Input image file stream
    :return: Array of bounding boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    # Formula to calculate the bounding box
    # conf = Confidence, Semakin Rendah Semakin Banyak Object yang Terdeteksi Tetapi Cenderung Salah Deteksi
    # iou = Intersection Over Union, Semakin Rendah Semakin Besar Kemungkinan Objek Yang Terdeteksi Tidak Tertabrak Satu Dengan Yang Lainnya.
    results = model.predict(image, conf=0.025, iou=0.2)
    result = results[0]
    output = []

    # Detected Object In Bounding Box
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        xAxis = x2 - x1
        yAxis = y2 - y1
        output.append(
            [x1, y1, x2, y2, result.names[class_id], prob, xAxis, yAxis])

    count = len(result.boxes)
    return output, count


if __name__ == "__main__":
    # Open the default web browser to the root URL
    webbrowser.open("http://localhost:5000")
    serve(app, host="0.0.0.0", port=5000)
