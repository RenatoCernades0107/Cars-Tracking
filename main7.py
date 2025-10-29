import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# ======================= Utilidades geométricas =======================


def iou(bb1: Sequence[float], bb2: Sequence[float]) -> float:
    """Calcula la intersección sobre unión de dos bboxes (x1,y1,x2,y2)."""

    xx1 = max(bb1[0], bb2[0])
    yy1 = max(bb1[1], bb2[1])
    xx2 = min(bb1[2], bb2[2])
    yy2 = min(bb1[3], bb2[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    a1 = max(0.0, (bb1[2] - bb1[0])) * max(0.0, (bb1[3] - bb1[1]))
    a2 = max(0.0, (bb2[2] - bb2[0])) * max(0.0, (bb2[3] - bb2[1]))
    return inter / (a1 + a2 - inter + 1e-6)


def bbox_to_z(bbox: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([cx, cy, s, r]).reshape(4, 1)


def z_to_bbox(state: np.ndarray) -> List[float]:
    cx, cy, s, r = state[0], state[1], state[2], state[3]
    w = np.sqrt(max(1e-6, s * r))
    h = s / max(1e-6, w)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [float(x1), float(y1), float(x2), float(y2)]


def nms_class_agnostic(
    detections: List[Tuple[Sequence[float], int, float]], iou_thr: float
) -> List[Tuple[Sequence[float], int, float]]:
    """NMS agnóstico de clase para evitar duplicados inter-clase."""

    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x[2], reverse=True)
    keep: List[Tuple[Sequence[float], int, float]] = []
    removed = [False] * len(detections)

    for i, (bbox_i, cls_i, score_i) in enumerate(detections):
        if removed[i]:
            continue
        keep.append((bbox_i, cls_i, score_i))
        for j in range(i + 1, len(detections)):
            if removed[j]:
                continue
            if iou(bbox_i, detections[j][0]) >= iou_thr:
                removed[j] = True

    return keep


# ======================= Tracker estilo SORT =======================


class KalmanTracker:
    _count = 0

    def __init__(self, bbox: Sequence[float], cls_id: int) -> None:
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = dt
        H = np.zeros((4, 8))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0

        self.kf.F = F
        self.kf.H = H
        self.kf.P *= 10.0
        self.kf.R *= 5.0
        self.kf.Q = np.eye(8) * 0.01

        self.kf.x[:4, 0] = bbox_to_z(bbox).flatten()
        self.id = KalmanTracker._count
        KalmanTracker._count += 1

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.cls_id = cls_id

    def predict(self) -> List[float]:
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return z_to_bbox(self.kf.x[:4, 0])

    def update(self, bbox: Sequence[float], cls_id: Optional[int] = None) -> None:
        z = bbox_to_z(bbox)
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if cls_id is not None:
            self.cls_id = cls_id

    def get_bbox(self) -> List[float]:
        return z_to_bbox(self.kf.x[:4, 0])


class SortManager:
    def __init__(self, iou_thr: float, max_age: int, min_hits: int) -> None:
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.min_hits = min_hits
        self.trks: List[KalmanTracker] = []

    def update(
        self, detections: List[Tuple[Sequence[float], int, float]]
    ) -> List[Dict[str, object]]:
        predicted = [trk.predict() for trk in self.trks]
        N, M = len(self.trks), len(detections)

        if N > 0 and M > 0:
            cost_matrix = np.zeros((N, M), dtype=np.float32)
            for i in range(N):
                for j in range(M):
                    cost_matrix[i, j] = 1.0 - iou(predicted[i], detections[j][0])
            row_idx, col_idx = linear_sum_assignment(cost_matrix)

            unmatched_trk = set(range(N))
            unmatched_det = set(range(M))
            matches: List[Tuple[int, int]] = []

            for r, c in zip(row_idx, col_idx):
                if 1.0 - cost_matrix[r, c] >= self.iou_thr:
                    matches.append((r, c))
                    unmatched_trk.discard(r)
                    unmatched_det.discard(c)

            for r, c in matches:
                bbox, cls_id, _ = detections[c]
                self.trks[r].update(bbox, cls_id)
        else:
            unmatched_trk = set(range(len(self.trks)))
            unmatched_det = set(range(len(detections)))

        for idx in list(unmatched_det):
            bbox, cls_id, _ = detections[idx]
            self.trks.append(KalmanTracker(bbox, cls_id))

        self.trks = [t for t in self.trks if t.time_since_update <= self.max_age]

        results: List[Dict[str, object]] = []
        for trk in self.trks:
            if trk.hits >= self.min_hits:
                results.append(
                    {
                        "id": trk.id,
                        "bbox": trk.get_bbox(),
                        "cls": trk.cls_id,
                        "tracker": trk,
                    }
                )
        return results


# ======================= Selección de líneas =======================


def interactive_select_lines(frame: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Permite seleccionar múltiples líneas sobre un frame.

    Controles:
        - Click izquierdo: agregar punto.
        - Click derecho: deshacer último punto o línea.
        - Tecla "c": limpiar todas las líneas.
        - Enter: confirmar selección.
        - Esc: cancelar (sin líneas).
    """

    window = "Selecciona líneas"
    cv2.namedWindow(window)
    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    pending: List[Tuple[int, int]] = []

    def draw_state() -> None:
        display = frame.copy()
        for (x1, y1), (x2, y2) in lines:
            cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(display, (x1, y1), 5, (0, 255, 255), -1)
            cv2.circle(display, (x2, y2), 5, (0, 255, 255), -1)
        for x, y in pending:
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (410, 125), (30, 30, 30), -1)
        display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)
        instructions = [
            "Click izq: puntos",
            "Click der: deshacer",
            "Enter: confirmar",
            "C: limpiar, Esc: cancelar",
        ]
        y = 35
        for text in instructions:
            cv2.putText(display, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25
        cv2.imshow(window, display)

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal pending
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x, y))
            if len(pending) == 2:
                pt1, pt2 = pending
                if pt1 != pt2:
                    lines.append((pt1, pt2))
                pending = []
        elif event == cv2.EVENT_RBUTTONDOWN:
            if pending:
                pending.pop()
            elif lines:
                lines.pop()
        draw_state()

    cv2.setMouseCallback(window, on_mouse)
    draw_state()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key == 27:  # Esc
            lines = []
            break
        if key in (ord("c"), ord("C")):
            lines = []
            pending = []
            draw_state()
        else:
            draw_state()

    cv2.destroyWindow(window)
    return lines


def load_lines_from_json(path: Path) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    data = json.loads(path.read_text())
    raw_lines = data.get("lines", [])
    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for item in raw_lines:
        if len(item) != 4:
            continue
        x1, y1, x2, y2 = map(int, item)
        lines.append(((x1, y1), (x2, y2)))
    return lines


def save_lines_to_json(path: Path, lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> None:
    payload = {
        "lines": [[x1, y1, x2, y2] for (x1, y1), (x2, y2) in lines],
        "comment": "Generado automáticamente por main7.py",
    }
    path.write_text(json.dumps(payload, indent=2))


# ======================= Utilidades de detección =======================


def draw_track(
    frame: np.ndarray,
    trk_data: Dict[str, object],
    class_names: Dict[int, str],
    color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = trk_data["bbox"]
    tid = trk_data["id"]
    name = class_names.get(trk_data["cls"], str(trk_data["cls"]))
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(
        frame,
        f"ID {tid} {name}",
        (int(x1), int(y1) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )
    cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)


def filter_by_roi(
    dets: List[Tuple[Sequence[float], int, float]], mask: np.ndarray
) -> List[Tuple[Sequence[float], int, float]]:
    filtered: List[Tuple[Sequence[float], int, float]] = []
    for bbox, cls_id, score in dets:
        x1, y1, x2, y2 = map(int, bbox)
        cx = max(0, min(mask.shape[1] - 1, (x1 + x2) // 2))
        cy = max(0, min(mask.shape[0] - 1, (y1 + y2) // 2))
        if mask[cy, cx]:
            filtered.append((bbox, cls_id, score))
    return filtered


# ======================= Ejecución principal =======================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conteo de vehiculos mejorado")
    parser.add_argument("--video", default="videos/videosample3.mp4", help="Ruta del video")
    parser.add_argument("--model", default="yolov8n.pt", help="Ruta del modelo YOLO")
    parser.add_argument("--frame-skip", type=int, default=1, help="Procesar 1 de cada N frames")
    parser.add_argument("--conf", type=float, default=0.35, help="Umbral de confianza YOLO")
    parser.add_argument("--iou", type=float, default=0.6, help="Umbral NMS interno YOLO")
    parser.add_argument("--nms", type=float, default=0.8, help="Umbral NMS agnostico")
    parser.add_argument("--auto", action="store_true", help="Modo sin lineas (conteo automatico)")
    parser.add_argument("--lines-json", type=str, help="Archivo JSON con lineas predefinidas")
    parser.add_argument(
        "--save-lines",
        type=str,
        help="Guardar las lineas seleccionadas en un JSON",
    )
    parser.add_argument("--output", type=str, help="Ruta del video de salida")
    parser.add_argument("--no-output", action="store_true", help="No guardar video procesado")
    parser.add_argument("--min-cross-hits", type=int, default=1, help="Golpes minimos antes de contar")
    parser.add_argument(
        "--cross-cooldown",
        type=float,
        default=1.0,
        help="Tiempo minimo (seg) entre cruces del mismo ID por linea",
    )
    parser.add_argument(
        "--roi-mask",
        type=str,
        help="Mascara (png) con ROI valida (valor > 0)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1200.0,
        help="Area minima de bbox para considerar deteccion",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("No se pudo abrir el video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    output_path = (
        Path(args.output)
        if args.output
        else video_path.with_name(video_path.stem + "_enhanced.mp4")
    )

    writer: Optional[cv2.VideoWriter] = None
    if not args.no_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[INFO] Guardando video procesado en: {output_path}")

    # ===== Máscara ROI opcional =====
    roi_mask: Optional[np.ndarray] = None
    if args.roi_mask:
        mask_path = Path(args.roi_mask)
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                print(f"[WARN] No se pudo cargar la máscara ROI: {mask_path}")
            elif mask_img.shape != (height, width):
                print("[WARN] La máscara ROI no coincide con el tamaño del video. Se ignora.")
            else:
                roi_mask = mask_img > 0
        else:
            print(f"[WARN] Ruta ROI inexistente: {mask_path}")

    # ===== Configuración de líneas =====
    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    if not args.auto:
        if args.lines_json:
            json_path = Path(args.lines_json)
            if json_path.exists():
                lines = load_lines_from_json(json_path)
                print(f"[INFO] {len(lines)} líneas cargadas desde {json_path}")
        if not lines:
            ret, first_frame = cap.read()
            if not ret:
                print("[ERROR] No se pudo leer el primer frame para seleccionar líneas")
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print(
                """
[INFO] Seleccion de lineas:
  1) Haz click izquierdo en el primer punto del segmento.
  2) Haz click izquierdo en el segundo punto (el tramo se pinta en rojo).
  3) Click derecho deshace el ultimo punto o linea.
  4) Presiona Enter para confirmar y salir. Presiona C para borrar todo.
  5) Repite para agregar mas lineas si lo necesitas.
""".strip()
            )
            lines = interactive_select_lines(first_frame)
            if args.save_lines and lines:
                save_lines_to_json(Path(args.save_lines), lines)
                print(f"[INFO] Líneas guardadas en {args.save_lines}")
        if not lines:
            print("[INFO] No se seleccionaron líneas. Cerrando.")
            return
    else:
        print("[INFO] Modo automático (sin líneas). Contando IDs únicos.")

    # ===== Cargar modelo =====
    model = YOLO(args.model)
    class_names = model.names
    vehicle_classes = {2, 3, 5, 7}

    sorter = SortManager(iou_thr=0.3, max_age=20, min_hits=2)

    line_counts = [0] * len(lines)
    counted_ids = [set() for _ in lines]
    last_side = [dict() for _ in lines]
    last_cross_time = [defaultdict(lambda: -np.inf) for _ in lines]
    unique_ids: set[int] = set()
    palette = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
    min_hits_for_cross = max(1, args.min_cross_hits)

    processed_frames = 0
    inference_times: List[float] = []
    loop_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frames += 1
        if processed_frames % args.frame_skip != 0:
            continue

        inf_start = time.time()
        preds = model.predict(
            frame,
            verbose=False,
            conf=args.conf,
            iou=args.iou,
            agnostic_nms=True,
        )[0]
        inf_end = time.time()
        inference_times.append(inf_end - inf_start)

        detections: List[Tuple[Sequence[float], int, float]] = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            for box, score, cls in zip(
                preds.boxes.xyxy.cpu().numpy(),
                preds.boxes.conf.cpu().numpy(),
                preds.boxes.cls.int().cpu().numpy(),
            ):
                if int(cls) not in vehicle_classes:
                    continue
                x1, y1, x2, y2 = box
                if (x2 - x1) * (y2 - y1) < args.min_area:
                    continue
                detections.append(([float(x1), float(y1), float(x2), float(y2)], int(cls), float(score)))

        detections = nms_class_agnostic(detections, args.nms)
        if roi_mask is not None:
            detections = filter_by_roi(detections, roi_mask)

        tracks = sorter.update(detections)

        current_time = processed_frames / fps

        if args.auto:
            for tr in tracks:
                unique_ids.add(tr["id"])
                draw_track(frame, tr, class_names, (0, 255, 0))
        else:
            for tr in tracks:
                tid = tr["id"]
                draw_track(frame, tr, class_names, (0, 255, 0))
                x1, y1, x2, y2 = tr["bbox"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                for idx, (A, B) in enumerate(lines):
                    side_now = np.sign((B[0] - A[0]) * (cy - A[1]) - (B[1] - A[1]) * (cx - A[0]))
                    side_prev = last_side[idx].get(tid, side_now)
                    last_side[idx][tid] = side_now

                    AB = np.array([B[0] - A[0], B[1] - A[1]])
                    AP = np.array([cx - A[0], cy - A[1]])
                    t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-6)
                    on_segment = 0.0 <= t <= 1.0

                    crossed = (
                        side_prev != 0
                        and side_now != 0
                        and side_prev != side_now
                        and on_segment
                        and tr["tracker"].hit_streak >= min_hits_for_cross
                    )

                    if not crossed:
                        continue

                    last_time = last_cross_time[idx][tid]
                    if current_time - last_time < args.cross_cooldown:
                        continue

                    counted_ids[idx].add(tid)
                    line_counts[idx] += 1
                    last_cross_time[idx][tid] = current_time

        # ===== Overlay =====
        overlay = frame.copy()
        panel_h = 40 + (30 * len(lines if not args.auto else [None]))
        cv2.rectangle(overlay, (20, 20), (280, 20 + panel_h), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        info_y = 45
        if args.auto:
            cv2.putText(
                frame,
                f"Vehiculos unicos: {len(unique_ids)}",
                (30, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            info_y += 30
            cv2.putText(
                frame,
                f"Tracks activos: {len(tracks)}",
                (30, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Conteo por linea:",
                (30, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            info_y += 30
            for idx, count in enumerate(line_counts):
                color = palette[idx % len(palette)]
                cv2.putText(
                    frame,
                    f"L{idx + 1}: {count}",
                    (35, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )
                info_y += 25
            total_lines = sum(line_counts)
            cv2.putText(
                frame,
                f"Total: {total_lines}",
                (30, info_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )

            for idx, (A, B) in enumerate(lines):
                color = palette[idx % len(palette)]
                cv2.line(frame, A, B, color, 3)
                cv2.circle(frame, A, 6, (255, 255, 255), -1)
                cv2.circle(frame, B, 6, (255, 255, 255), -1)

        cv2.imshow("YOLO + Kalman tracking (mejorado)", frame)
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - loop_start
    mean_inf = (sum(inference_times) / len(inference_times)) if inference_times else 0.0

    print("\n==============================")
    if args.auto:
        print(f"Vehiculos unicos detectados: {len(unique_ids)}")
    else:
        for idx, count in enumerate(line_counts, 1):
            print(f"Conteo linea L{idx}: {count}")
        print(f"Total (suma de lineas): {sum(line_counts)}")
    print(f"Frames procesados: {processed_frames}")
    print(f"Tiempo total: {elapsed:.2f} s")
    if mean_inf > 0:
        print(f"Tiempo promedio inferencia: {mean_inf*1000:.1f} ms")
    if writer is not None:
        print(f"Video guardado en: {output_path}")
    print("==============================\n")


if __name__ == "__main__":
    main()


