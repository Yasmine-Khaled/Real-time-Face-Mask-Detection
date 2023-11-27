import cv2
import numpy as np
from keras.models import load_model

model = load_model("resnet101v2.h5")

results = {0: 'With Mask', 1: 'Without Mask'}
color = {0: (0, 255, 0), 1: (0, 0, 255)}

rect_size = 4
cap = cv2.VideoCapture(0)

# Use a more robust face detector, such as MTCNN or Dlib
haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    (rval, im) = cap.read()
    im = cv2.flip(im, 1, 1)

    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))

    # Detect faces using the cascade classifier
    faces = haarcascade.detectMultiScale(rerect_size)

    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        face = im[y:y + h, x:x + w]

        # Resize the face to match the model's input size (256x256)
        face = cv2.resize(face, (256, 256))

        # Normalize and reshape the face for model input
        normalized = face / 255.0
        reshaped = np.reshape(normalized, (1, 256, 256, 3))
        reshaped = np.vstack([reshaped])

        # Make predictions using the loaded model
        result = model.predict(reshaped)

        # Get the predicted label
        label = np.argmax(result, axis=1)[0]

        # Draw rectangles and text on the image
        cv2.rectangle(im, (x, y), (x + w, y + h), color[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), color[label], -1)
        cv2.putText(im, results[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the processed image
    cv2.imshow('LIVE', im)

    # Break the loop when the window is closed
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' key or Esc key
        break

# Release the video capture resources
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
