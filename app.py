from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.applications import MobileNetV2           

app = Flask(__name__,template_folder='templates')

project_root = os.path.abspath(os.path.dirname(__file__))
# Step 1: Define the paths to your dataset folders
conjunctivitis_dir = os.path.join(project_root, "output", "conjunctivitis")
non_conjunctivitis_dir = os.path.join(project_root, "output", "normal")

# Step 2: Create lists to store image file paths and their corresponding labels
image_paths = []
labels = []

# Define label mappings for each class
class_labels = {
    "conjunctivitis": 0,
    "non_conjunctivitis": 1
}

# Loop through the conjunctivitis folder
for image_file in os.listdir(conjunctivitis_dir):
    image_path = os.path.join(conjunctivitis_dir, image_file)
    image_paths.append(image_path)
    labels.append(class_labels["conjunctivitis"])

# Loop through the non-conjunctivitis folder
for image_file in os.listdir(non_conjunctivitis_dir):
    image_path = os.path.join(non_conjunctivitis_dir, image_file)
    image_paths.append(image_path)
    labels.append(class_labels["non_conjunctivitis"])

    # Step 3: Resize images to a consistent dimension
target_size = (224, 224)
resized_images = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    resized_images.append(image)

resized_images = np.array(resized_images)

# Step 4: Normalize pixel values
normalized_images = resized_images / 255.0

# Step 5: Check the shapes of the resulting arrays
print("Shape of resized_images:", resized_images.shape)
print("Shape of normalized_images:", normalized_images.shape)

# Step 6: Split your dataset into training, validation, and testing sets
# Define the ratio for splitting (e.g., 70% train, 15% validation, 15% test)
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Split the dataset into training (70%) and temporary (30%)
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(
    normalized_images, labels, test_size=(1 - train_ratio), random_state=42)

# Split the temporary dataset into validation (15%) and testing (15%)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=test_ratio / (test_ratio + validation_ratio), random_state=42)

# Ensure that the data is properly split
print("Number of samples in training set:", len(X_train_temp))
print("Number of samples in validation set:", len(X_validation))
print("Number of samples in testing set:", len(X_test))


# Step 7: Model Selection and Training

# Define the architecture of your model
def create_model(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Define input shape and number of classes (2 for conjunctivitis and non-conjunctivitis)


input_shape = (224, 224, 3)
num_classes = 2

# Create the model
model = create_model(input_shape, num_classes)

# Compile the model with an appropriate optimizer, loss function, and evaluation metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' for one-hot encoded labels
              metrics=['accuracy'])

# Define batch size and number of training epochs
batch_size = 32
epochs = 10

# Convert labels to NumPy arrays with the correct data type
y_train_temp = np.array(y_train_temp, dtype=np.int64)
y_validation = np.array(y_validation, dtype=np.int64)
y_test = np.array(y_test, dtype=np.int64)

# Train the model using the training and validation data
history = model.fit(X_train_temp, y_train_temp,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_validation, y_validation))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Now, you have trained and evaluated your model. You can fine-tune hyperparameters and monitor performance as needed.
# Save the trained model to a file
# Save the trained model in the native Keras format
model_path = "conjunctivitis_model.keras"
model.save(model_path)

def load_model():
    global model
    model = keras.models.load_model(model_path)


import cv2
import numpy as np
from tensorflow import keras

# Load the saved model

# Define label mappings
class_labels = {
    0: "conjunctivitis",
    1: "non_conjunctivitis"
}


def preprocess_image(image_path):
    # Load and preprocess the input image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def predict_conjunctivitis(image_path):
    # Preprocess the input image
    input_image = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(input_image)

    # Get the predicted class label
    predicted_class = np.argmax(prediction)

    # Get the class name
    class_name = class_labels[predicted_class]

    return class_name


def capture_from_camera():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Test")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Test", frame)
        k = cv2.waitKey(1) & 0xFF  # Wait for a key event

        if k == 27:  # "Esc" key (key code 27)
            print("Closing...")
            break
        elif k == 32:  # "Space" key (key code 32) to capture an image
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()
    return img_name

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/options', methods=['POST'])
def options():
    name = request.form['name']
    age = request.form['age']
    return render_template('options.html', name=name, age=age)

@app.route('/camera')
def camera():

    return render_template('camera.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/capture')
def capture():
    image_path = capture_from_camera()
    result = predict_conjunctivitis(image_path)
    if result == "conjunctivitis":
        msg = "You have been diagnosed with conjunctivitis 😟"
    else:
        msg = "You have not been diagnosed with conjunctivitis 😃"
    return render_template('result.html', image_path=image_path,result = msg)

    # return render_template('result.html', image_path=image_path, result = result)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        image_path = file.filename
        file.save(image_path)
        result = predict_conjunctivitis(image_path)
        msg = " "
        if result == "conjunctivitis":
            msg = "You have been diagnosed with conjunctivitis 😟"
        else:
            msg = "You have not been diagnosed with conjunctivitis 😃"
    return render_template('result.html', image_path=image_path,result = msg)
    # return render_template('result.html', image_path=image_path, result=result)

@app.route('/precautions')
def precautions():
    return render_template('precautions.html')


if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5500)  
