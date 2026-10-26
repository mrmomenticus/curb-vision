from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.predict(
    source=Path(__file__).with_name("gr2sobpl.jpg"),
    classes=[2],
)

results[0].save(filename="detected.jpg")
results[0].show()

for box in results[0].boxes:
    class_id = int(box.cls.item())
    name = model.names[class_id]
    confidence = box.conf.item()
    coordinates = box.xyxy[0].tolist()

    print(name, round(confidence, 2), coordinates)
