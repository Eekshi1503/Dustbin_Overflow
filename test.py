from ultralytics import YOLO
import cv2

model = YOLO("best_yolov8n.onnx")

image_path = "Dustbin_overflow.v1i.yolov8/test/images/1652767312573_jpg.rf.b4210004a90725213247ee83ebe44c02.jpg"  
results = model.predict(source=image_path, conf=0.5)

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

print("Final Status:", final_label)
print("Confidence:", best_conf)

# Show image
annotated = results[0].plot()
cv2.imshow("Dustbin Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()