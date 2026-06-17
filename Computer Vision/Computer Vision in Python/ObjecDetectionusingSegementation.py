from ultralytics import YOLO
import cv2
import torch
import numpy as np

# ==========================
# GPU CHECK
# ==========================
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    device = "cuda"
else:
    device = "cpu"

# ==========================
# LOAD SEGMENTATION MODEL
# ==========================
model = YOLO("yolov8m-seg.pt")

# ==========================
# VIDEO
# ==========================
video_path = r"C:\Users\HP\Desktop\Adi\Progrmming\python\AI GT\dataset\Traffica.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Unable to open video")

# First frame
ret, frame = cap.read()

if not ret:
    raise ValueError("Cannot read first frame")

h, w = frame.shape[:2]

# Reset to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ==========================
# OUTPUT VIDEO
# ==========================
out = cv2.VideoWriter("segmentation_counting_outputN.mp4",cv2.VideoWriter_fourcc(*"mp4v"),25,(w, h))

# ==========================
# COUNTING LINE
# ==========================
line_y = h // 2

# ==========================
# CLASSES TO KEEP
# ==========================
allowed_classes = [
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "person"
]

# ==========================
# COUNTERS
# ==========================
counted_ids = set()

car_count = 0
bus_count = 0
truck_count = 0
motorcycle_count = 0
bicycle_count = 0
person_count = 0

frame_number = 0

# ==========================
# MAIN LOOP
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # ==========================
    # TRACK + SEGMENTATION
    # ==========================
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        device=device,
        verbose=False
    )

    vehicle_area = 0

    # ==========================
    # PROCESS RESULTS
    # ==========================
    for r in results:

        if r.boxes is None:
            continue

        masks = None

        if r.masks is not None:
            masks = r.masks.xy

        for i, box in enumerate(r.boxes):

            if box.id is None:
                continue

            track_id = int(box.id.item())

            cls = int(box.cls.item())

            label = model.names[cls]

            if label not in allowed_classes:
                continue

            conf = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # ==================================
            # DRAW SEGMENTATION MASK
            # ==================================
            if masks is not None and i < len(masks):

                segment = masks[i].astype(np.int32)

                overlay = frame.copy()

                cv2.fillPoly(overlay, [segment], (0, 255, 0))

                frame = cv2.addWeighted( overlay, 0.25, frame, 0.75, 0)

                cv2.polylines(frame, [segment], True, (0, 255, 0), 2)

                # ==========================
                # SEGMENTATION AREA
                # ==========================
                temp_mask = np.zeros((h, w), dtype=np.uint8)

                cv2.fillPoly(temp_mask, [segment], 255)

                vehicle_area += cv2.countNonZero(temp_mask)

                # ==========================
                # SEGMENT CENTROID
                # ==========================
                M = cv2.moments(segment)

                if M["m00"] != 0:

                    cx = int(M["m10"] / M["m00"])

                    cy = int(M["m01"] / M["m00"])

                else:

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

            else:

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

            # ==================================
            # DRAW CENTROID
            # ==================================
            cv2.circle(frame,(cx, cy),4,(0, 0, 255),-1)

            # ==================================
            # DRAW LABEL
            # ==================================
            cv2.putText(frame,f"{label} {track_id}",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)

            # ==================================
            # COUNT OBJECTS
            # ==================================
            if line_y - 5 < cy < line_y + 5:

                if track_id not in counted_ids:

                    counted_ids.add(track_id)

                    if label == "car":
                        car_count += 1

                    elif label == "bus":
                        bus_count += 1

                    elif label == "truck":
                        truck_count += 1

                    elif label == "motorcycle":
                        motorcycle_count += 1

                    elif label == "bicycle":
                        bicycle_count += 1

                    elif label == "person":
                        person_count += 1

    # ==========================
    # TRAFFIC DENSITY
    # ==========================
    total_area = h * w

    density = (vehicle_area / total_area) * 100

    # ==========================
    # DRAW COUNTING LINE
    # ==========================
    cv2.line(frame,(0, line_y),(w, line_y),(255, 0, 0),3)

    # ==========================
    # DISPLAY COUNTS
    # ==========================
    cv2.putText(frame,f"Cars: {car_count}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)

    cv2.putText(frame,f"Buses: {bus_count}",(20, 80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)

    cv2.putText(frame,f"Trucks: {truck_count}",(20, 120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)

    cv2.putText(frame,f"Motorcycle: {motorcycle_count}",(20, 160),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)

    cv2.putText(frame,f"Bicycle: {bicycle_count}",(20, 200),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)

    cv2.putText(frame,f"Person: {person_count}",(20, 240),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)

    cv2.putText(frame,f"Density: {density:.2f} %",(20, 280),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 255),2)

    cv2.putText(frame,f"Frame: {frame_number}",(20, 320),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 0),2)

    # ==========================
    # SAVE OUTPUT
    # ==========================
    out.write(frame)

    cv2.imshow("AI Traffic System - Segmentation",frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
out.release()

cv2.destroyAllWindows()

print("\n========== FINAL COUNTS ==========")
print("Cars       :", car_count)
print("Buses      :", bus_count)
print("Trucks     :", truck_count)
print("Motorcycle :", motorcycle_count)
print("Bicycle    :", bicycle_count)
print("Persons    :", person_count)
