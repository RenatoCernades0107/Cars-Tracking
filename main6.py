import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from setline import set_line_from_video
import time

# ======= Utilidades geométricas =======
def iou(bb1, bb2):
    # boxes [x1,y1,x2,y2]
    xx1 = max(bb1[0], bb2[0]); yy1 = max(bb1[1], bb2[1])
    xx2 = min(bb1[2], bb2[2]); yy2 = min(bb1[3], bb2[3])
    w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
    inter = w * h
    a1 = max(0.0, (bb1[2]-bb1[0])) * max(0.0, (bb1[3]-bb1[1]))
    a2 = max(0.0, (bb2[2]-bb2[0])) * max(0.0, (bb2[3]-bb2[1]))
    return inter / (a1 + a2 - inter + 1e-6)

def bbox_to_z(b):
    x1,y1,x2,y2 = b
    w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
    cx = x1 + w/2.0; cy = y1 + h/2.0
    s = w * h; r = w / h
    return np.array([cx, cy, s, r]).reshape(4,1)

def z_to_bbox(x):
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    w = np.sqrt(max(1e-6, s*r)); h = s / max(1e-6, w)
    x1 = cx - w/2.0; y1 = cy - h/2.0
    x2 = cx + w/2.0; y2 = cy + h/2.0
    return [float(x1), float(y1), float(x2), float(y2)]

def point_side(p, a, b):
    # signo del lado respecto a la recta AB (producto cruzado 2D)
    return np.sign((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]))

def nms_class_agnostic(dets, iou_thr=0.8):
    """
    dets: list of (bbox [x1,y1,x2,y2], cls_id, score)
    Devuelve una lista filtrada por NMS (agnóstico de clase).
    """
    if not dets: 
        return []
    # ordenar por score desc
    dets = sorted(dets, key=lambda x: x[2], reverse=True)
    keep = []
    used = [False]*len(dets)
    for i in range(len(dets)):
        if used[i]:
            continue
        keep.append(dets[i])
        bi = dets[i][0]
        for j in range(i+1, len(dets)):
            if used[j]:
                continue
            bj = dets[j][0]
            if iou(bi, bj) >= iou_thr:
                used[j] = True
    return keep

# ======= Kalman Tracker tipo SORT =======
class KalmanTracker:
    _count = 0
    def __init__(self, bbox, cls_id):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        F = np.eye(8)
        for i in range(4): F[i, i+4] = dt
        H = np.zeros((4,8)); H[0,0]=H[1,1]=H[2,2]=H[3,3]=1.0
        self.kf.F = F; self.kf.H = H
        self.kf.P *= 10.0            # incertidumbre inicial
        self.kf.R *= 5.0             # ruido de medición (ajustable)
        self.kf.Q = np.eye(8)*0.01   # ruido de proceso (ajustable)

        self.kf.x[:4,0] = bbox_to_z(bbox).flatten()
        self.id = KalmanTracker._count; KalmanTracker._count += 1
        self.time_since_update = 0
        self.hits = 1; self.hit_streak = 1; self.age = 0
        self.cls_id = cls_id         # clase “vehículo” asociada
        self._last_centroid = None   # para cruce de línea

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0: self.hit_streak = 0
        self.time_since_update += 1
        return z_to_bbox(self.kf.x[:4,0])

    def update(self, bbox, cls_id=None):
        z = bbox_to_z(bbox)
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1; self.hit_streak += 1
        if cls_id is not None: self.cls_id = cls_id

    def get_bbox(self):
        return z_to_bbox(self.kf.x[:4,0])

    def centroid(self):
        x1,y1,x2,y2 = self.get_bbox()
        return ((x1+x2)/2.0, (y1+y2)/2.0)

# ======= Gestor de trackers (asignación Hungarian + IoU gating) =======
class SortManager:
    def __init__(self, iou_thr=0.3, max_age=20, min_hits=2):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.min_hits = min_hits
        self.trks = []

    def update(self, detections):
        """
        detections: list of (bbox[x1,y1,x2,y2], cls_id, score)
        return: list of dicts {id,bbox,cls}
        """
        # 1) predecir todos
        predicted = [trk.predict() for trk in self.trks]

        # 2) matriz de coste 1-IoU
        N, M = len(self.trks), len(detections)
        if N>0 and M>0:
            cost = np.zeros((N,M), dtype=np.float32)
            for i in range(N):
                for j in range(M):
                    cost[i,j] = 1.0 - iou(predicted[i], detections[j][0])
            row_idx, col_idx = linear_sum_assignment(cost)
            unmatched_trk = set(range(N)); unmatched_det = set(range(M))
            matches = []
            for r,c in zip(row_idx, col_idx):
                if 1.0 - cost[r,c] >= self.iou_thr:
                    matches.append((r,c))
                    unmatched_trk.discard(r); unmatched_det.discard(c)
            # 3) actualizar emparejados
            for r,c in matches:
                bb, cls_id, _ = detections[c]
                self.trks[r].update(bb, cls_id)
        else:
            unmatched_trk = set(range(len(self.trks)))
            unmatched_det = set(range(len(detections)))

        # 4) crear trackers nuevos para no emparejados
        for c in list(unmatched_det):
            bb, cls_id, _ = detections[c]
            self.trks.append(KalmanTracker(bb, cls_id))

        # 5) eliminar viejos
        self.trks = [t for t in self.trks if t.time_since_update <= self.max_age]

        # 6) salida confirmada
        out = []
        for t in self.trks:
            # O al menos dos veces actualizado
            if t.hits >= self.min_hits:
                out.append({"id": t.id, "bbox": t.get_bbox(), "cls": t.cls_id, "tracker": t})
        return out

def main():
    start_time = time.time()

    video_path = "videos/videosample3.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video"); 
        return

    # === Configuración del guardado de video ===
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = video_path.split('/')[-1].split('.')[0] + "_output.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"[INFO] Guardando video procesado en: {output_path}")

    # ===== Selección múltiple de líneas =====
    print("[INFO] Dibuja una línea (2 clicks) y presiona ENTER. Cierra sin dibujar para terminar.")
    lines = []
    while True:
        try:
            start, end, points = set_line_from_video(video_path)
            if points is None:
                break
            A, B = points
            A = (int(A[0]), int(A[1])); B = (int(B[0]), int(B[1]))
            if A != B:
                lines.append((A, B))
                print(f"[INFO] Línea {len(lines)} registrada: {A} -> {B}")
            else:
                print("[WARN] Puntos iguales; línea ignorada.")
        except Exception:
            print("[INFO] Selección de líneas finalizada.")
            break

        resp = input("¿Agregar otra línea? (s/n): ").strip().lower()
        if resp != "s":
            break

    if not lines:
        print("No se seleccionaron líneas. Saliendo.")
        cap.release(); out.release()
        return

    # ===== Configuración YOLO =====
    model = YOLO("yolov8n.pt")
    class_names = model.names
    VEHICLE_CLS = {2,3,5,7}
    CONF_THR = 0.35
    FRAME_SKIP = 1

    sorter = SortManager(iou_thr=0.3, max_age=20, min_hits=2)

    num_lines = len(lines)
    line_counts = [0] * num_lines
    counted_ids = [set() for _ in range(num_lines)]
    last_side   = [dict() for _ in range(num_lines)]
    palette = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0)]

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_i += 1
        if frame_i % FRAME_SKIP != 0:
            continue

        # ===== Detección YOLO =====
        preds = model.predict(frame, verbose=False, conf=CONF_THR, iou=0.6, agnostic_nms=True)[0]
        dets = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            for box, score, cls in zip(preds.boxes.xyxy.cpu().numpy(),
                                       preds.boxes.conf.cpu().numpy(),
                                       preds.boxes.cls.int().cpu().numpy()):
                if int(cls) in VEHICLE_CLS:
                    x1,y1,x2,y2 = box.astype(int)
                    dets.append(([x1,y1,x2,y2], int(cls), float(score)))

        dets = nms_class_agnostic(dets, iou_thr=0.8)
        tracks = sorter.update(dets)

        # ===== Conteo por línea =====
        for tr in tracks:
            tid = tr["id"]; x1,y1,x2,y2 = tr["bbox"]
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            for i, (A, B) in enumerate(lines):
                side_now  = np.sign((B[0]-A[0])*(cy-A[1]) - (B[1]-A[1])*(cx-A[0]))
                side_prev = last_side[i].get(tid, side_now)
                last_side[i][tid] = side_now
                AB = np.array([B[0]-A[0], B[1]-A[1]])
                AP = np.array([cx-A[0], cy-A[1]])
                t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-6)
                on_segment = (0.0 <= t <= 1.0)
                crossed = (side_prev != 0) and (side_now != 0) and (side_prev != side_now) and on_segment
                if crossed and tid not in counted_ids[i]:
                    counted_ids[i].add(tid)
                    line_counts[i] += 1

        # ===== Dibujo =====
        # Dibujar tracks
        for tr in tracks:
            tid = tr["id"]; x1,y1,x2,y2 = tr["bbox"]
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            name = class_names.get(tr["cls"], str(tr["cls"]))
            cv2.putText(frame, f"ID {tid} {name}", (int(x1), int(y1)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.circle(frame, (int(cx),int(cy)), 3, (0,0,255), -1)

        # Dibujar líneas
        for i, (A, B) in enumerate(lines):
            color = palette[i % len(palette)]
            cv2.line(frame, A, B, color, 3)
            cv2.circle(frame, A, 6, (0, 255, 255), -1)
            cv2.circle(frame, B, 6, (0, 255, 255), -1)

        # === Tabla resumen (esquina superior izquierda) ===
        overlay = frame.copy()
        table_x, table_y = 20, 20
        table_w, table_h = 260, 40 + 30 * num_lines
        cv2.rectangle(overlay, (table_x, table_y), (table_x+table_w, table_y+table_h), (40,40,40), -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

        y_text = table_y + 35
        cv2.putText(frame, "Conteo por linea:", (table_x + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        for i, c in enumerate(line_counts):
            color = palette[i % len(palette)]
            y_text += 25
            cv2.putText(frame, f"L{i+1}: {c}", (table_x + 25, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        total_all = sum(line_counts)
        y_text += 35
        cv2.putText(frame, f"Total: {total_all}", (table_x + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        out.write(frame)
        cv2.imshow("YOLO + Kalman tracking (multi-line count)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === Finalización ===
    cap.release(); out.release(); cv2.destroyAllWindows()
    elapsed = time.time() - start_time

    print("\n==============================")
    for i, c in enumerate(line_counts, 1):
        print(f"Conteo línea L{i}: {c}")
    print(f"Total (suma de líneas): {sum(line_counts)}")
    print(f"Tiempo total de ejecución: {elapsed:.2f} segundos")
    print(f"Video guardado exitosamente en '{output_path}'")
    print("==============================\n")

if __name__ == "__main__":
    main()
