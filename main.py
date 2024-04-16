import cv2
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

def is_bbox_abnormal(bbox, max_area):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return area > max_area
SCALE_THRESHOLD = 0.2
def is_bbox_too_large(bbox, previous_bbox):
    x1, y1, x2, y2 = bbox
    prev_x1, prev_y1, prev_x2, prev_y2 = previous_bbox
    current_area = (x2 - x1) * (y2 - y1)
    previous_area = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
    area_change = current_area / previous_area
    return area_change > (1 + SCALE_THRESHOLD) or area_change < (1 - SCALE_THRESHOLD)

def counter_sheep(video_path, tracking_class=18, conf_threshold=0.5, line_x=50):
    # Khởi tạo biến đếm số cừu và tập hợp để lưu trữ id cừu đã đếm
    sheep_count = 0
    counted_sheep_ids = set()
    
    tracker = DeepSort(max_age=75,nms_max_overlap=0.2)

    device = "cpu"
    model = DetectMultiBackend(weights="C:/DOANCPVS/YOLOV9/weights/yolov9-e-converted.pt", device=device, fuse=True)
    model = AutoShape(model)

    # Tải tên các lớp
    with open("C:/DOANCPVS/YOLOV9/data/class.names") as f:
        class_names = f.read().strip().split('\n')
    colors = np.random.randint(0, 255, size=(len(class_names), 3))
    previous_bboxes = []
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_bbox_area = frame_width * frame_height * 0.25
    # Đọc từng khung hình từ video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện các đối tượng sử dụng YOLOv9
        results = model(frame)
        detections = []

        for detection in results.pred[0]:
            label, confidence, bbox = detection[5], detection[4], detection[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            # Kiểm tra xem đối tượng có phải là cừu và vượt qua ngưỡng tin cậy không
            if tracking_class is None:
                if confidence < conf_threshold:
                    continue
            else:
                if class_id != tracking_class or confidence < conf_threshold:
                    continue

            detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        # Cập nhật các đối tượng sử dụng DeepSort
        tracks = tracker.update_tracks(detections, frame=frame)

        # Vẽ các khung hình và id
        for track in tracks:
            if track.is_confirmed() and track.time_since_update <4:
                track_id = track.track_id

                # Lấy toạ độ, class_id để vẽ lên hình ảnh
                ltrb = track.to_ltrb()
                if is_bbox_abnormal(ltrb, max_bbox_area): # loai bo cac bbox to dot ngot
                    continue
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                if track_id in previous_bboxes and is_bbox_too_large((x1, y1, x2, y2), previous_bboxes[track_id]):
                    continue
                color = colors[class_id]
                B, G, R = map(int,color)

                label = "{}-{}".format(class_names[class_id], track_id)

                # Kiểm tra xem đối tượng có vượt qua đường thẳng và chưa được đếm chưa
                if x1 < line_x and x2 > line_x and track_id not in counted_sheep_ids:
                    sheep_count += 1
                    counted_sheep_ids.add(track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), color.tolist(), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Vẽ đường thẳng bên trái của khung hình
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 0), 2)

        # Hiển thị số cừu đã đếm trên video
        cv2.putText(frame, f"counter sheep: {sheep_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị video
        cv2.imshow("VIDEO", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Gọi hàm count_sheep để chạy đoạn mã
counter_sheep("C:/DOANCPVS/YOLOV9/data.ext/video.mp4", tracking_class=18, conf_threshold=0.5, line_x=50)
