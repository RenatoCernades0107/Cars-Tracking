# Cars Tracking

## Comparación: Kalman Filter vs YOLO tracking

Este proyecto compara dos enfoques de tracking de vehículos:
- **YOLO Tracking integrado** (`main.py`)
- **Kalman Filter personalizado** (`main2.py`, `main3.py`)

## Archivos principales

### 📋 Tabla comparativa

| Archivo | Tracking | Modelo | Video | Clases detectadas | Características |
|---------|----------|--------|-------|-------------------|-----------------|
| `main.py` | YOLO built-in tracking | `yolo11l.pt` | `videosample1.mp4` | [1,2,3,5,6,7] | Rápido, tracking automático de YOLO |
| `main2.py` | Kalman Filter (SORT) | `yolov8n.pt` | `videosample1.mp4` | {2,3,5,7} | Control total, mejor persistencia |
| `main3.py` | Kalman Filter (SORT) | `yolov8n.pt` | `videosample2.mp4` | {2,3,5,7} | Igual que main2 + filtro NMS extra |
| `main4.py` | Kalman Filter (SORT) | `yolov8n.pt` | `videosample1.mp4` | {2,3,5,7} | **100% AUTOMÁTICO** - Sin línea manual |

### 🔍 Diferencias técnicas

**main.py (YOLO Tracking):**
- Usa `model.track()` con `persist=True`
- Más simple pero menos control
- Frame skipping de 10

**main2.py / main3.py (Kalman Filter):**
- Implementación SORT personalizada
- Kalman Filter 8D (posición + velocidad)
- Hungarian algorithm para asignación
- IoU threshold = 0.3
- max_age = 20 frames
- min_hits = 2
- main3.py tiene NMS agnóstico adicional (iou_thr=0.8)

**main4.py (Kalman Filter - AUTOMÁTICO):**
- ✅ **No requiere trazar línea con el mouse**
- Cuenta todos los vehículos únicos que aparecen
- Basado en main3.py (Kalman + NMS)
- Ideal para pruebas rápidas sin intervención manual
- Muestra: Total único y vehículos actualmente rastreados

### 📏 Sistema de línea de conteo

**main.py, main2.py, main3.py (con línea manual):** 
- Llaman a `set_line_from_video(video_path)` al inicio
- Requieren **2 clics** con el mouse para definir la línea
- La línea se usa para contar cruces de vehículos

**main4.py (sin línea - automático):**
- ✅ **No requiere clics ni línea**
- Cuenta automáticamente todos los vehículos únicos detectados
- Ideal para pruebas rápidas

**Detección de cruce:**
- **main.py**: Proyección perpendicular (producto cruzado)
- **main2.py / main3.py**: Cambio de lado + validación en segmento [A,B]
- **main4.py**: Sin línea - cuenta IDs únicos directamente

### 🚀 Uso rápido

**Modo automático (recomendado para pruebas):**
```bash
python main4.py
```
- No requiere interacción, se ejecuta inmediatamente
- Presiona `q` para salir

**Modo con línea de conteo:**
```bash
python main.py    # YOLO tracking
python main2.py   # Kalman Filter
python main3.py   # Kalman Filter + NMS
```
- Al iniciar, haz clic en 2 puntos para trazar la línea
- Presiona `q` o `x` para salir

### TODO:

- ✅ Implementar Kalman Filter (main2.py, main3.py)
- ✅ Modo automático sin línea (main4.py)
- ⏳ Exportar video procesado
- ⏳ Realizar comparación de rendimiento
