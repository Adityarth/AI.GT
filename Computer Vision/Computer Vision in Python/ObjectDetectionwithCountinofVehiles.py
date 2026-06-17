from ultralytics import YOLO
import cv2
import torch

#Checks GPU is available to use or not
print(torch.cuda.is_available())  
print(torch.cuda.get_device_name(0))

# Load model
model = YOLO("yolov8m.pt")

video_path = r"C:\Users\HP\Desktop\Adi\Progrmming\python\AI GT\dataset\Traffica.mp4"
cap = cv2.VideoCapture(video_path)
fr = 0
# Read first frame
ret, frame = cap.read()
if not ret:
    raise ValueError("Video not loading")

h, w = frame.shape[:2]

# Output video
out = cv2.VideoWriter(
    "dataset/video object detection with counting5.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    25,
    (w, h)\
      
)

# Line position (horizontal line)
line_y = h // 2

# Allowed classes (traffic)
allowed_classes = ["car", "bus", "truck", "person", "bicycle", "motorcycle"]

# Tracking memory
counted_ids = set()

# Counters
car_count = 0
bus_count = 0
truck_count = 0
people_count = 0
bicycle_count = 0
motorcycle_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 TRACKING ENABLED
    results = model.track(frame, persist=True, device = "cuda")
    
    fr = fr+1

    for r in results:
        for box in r.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # crossing point of object
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Draw center
            cv2.circle(frame, (cx, cy), 2, (0,0,255), -1)

            # 🔥 COUNTING LOGIC
            if cy > line_y - 5 and cy < line_y + 5:
                if track_id not in counted_ids: 
                    counted_ids.add(track_id)

                    if label == "car":
                        car_count += 1
                    elif label == "bus":
                        bus_count += 1
                    elif label == "truck":
                        truck_count += 1
                    elif label == "people":
                        people_count += 1
                    elif label == "bicycle":
                        bicycle_count += 1
                    elif label == "motorcycle":
                        motorcycle_count += 1
    
    print (f"Frame: {fr}")

    # Draw counting line
    cv2.line(frame, (0, line_y), (w, line_y), (255,0,0), 2)

    # Show counts
    cv2.putText(frame, f"Cars: {car_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
    cv2.putText(frame, f"Buses: {bus_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
    cv2.putText(frame, f"Trucks: {truck_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
    cv2.putText(frame, f"people: {people_count}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
    cv2.putText(frame, f"bicycle: {bicycle_count}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
    cv2.putText(frame, f"motorcycle: {motorcycle_count}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

    out.write(frame)
    
    
    cv2.imshow("Traffic System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
