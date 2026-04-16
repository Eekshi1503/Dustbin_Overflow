from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best_yolov8n.onnx")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.75, verbose=False)

    final_label = "No Detection"
    best_conf = 0.0

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if confidence > best_conf:
                best_conf = confidence
                final_label = class_name

    # Draw predictions
    annotated = results[0].plot()

    # Display label
    cv2.putText(
        annotated,
        f"{final_label} ({best_conf:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Dustbin Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()