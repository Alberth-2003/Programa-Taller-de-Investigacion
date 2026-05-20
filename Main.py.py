# =============================================================================
# OE1 — Pipeline completo: 6 cámaras simuladas → dataset sintético
# Tesis: Optimización ITS Metropolitano de Lima
# =============================================================================
# CÓMO EJECUTAR:
#   cd "C:\Users\Alberth\Desktop\TALLER DE INVESTIGACION\Programa CCTV ATU"
#   python Prueba1.py
#
# SALIDA:
#   - 6 CSVs individuales (uno por cámara)
#   - 1 CSV consolidado con todos los datos juntos
#   - Resumen final en consola con métricas para tu tesis
# =============================================================================

import cv2
import csv
import time
import os
from ultralytics import YOLO
from datetime import datetime

# ── Configuración global ───────────────────────────────────────────────────────
MODELO      = 'yolov8n.pt'
CONFIANZA   = 0.3
CLASES      = [0]        # 0 = persona
SALTO       = 2          # procesar 1 de cada 2 frames
CARPETA     = r"C:\Users\Alberth\Desktop\TALLER DE INVESTIGACION\PROGRAMA TESIS"
MOSTRAR     = True       # False para correr más rápido sin ventana de video

# ── Definición de las 6 cámaras ───────────────────────────────────────────────
# Formato: (nombre_archivo, estacion, camara)
CAMARAS = [
    ("Video01_Caqueta_A1.mp4",        "Caqueta", "CAM-A1"),
    ("Video02_Caqueta_A2.mp4",        "Caqueta", "CAM-A2"),
    ("video02_UNI_A3.mp4",            "Caqueta", "CAM-A3"),  # proxy
    ("Video01_Tacna_B1.mp4",          "Tacna",   "CAM-B1"),
    ("Video03_EstadioNacional_B2.mp4", "Tacna",  "CAM-B2"),  # proxy
    ("Video03_Aramburu_B3.mp4",        "Tacna",  "CAM-B3"),  # proxy
]
# Segundos a saltar al inicio de cada video (para evitar partes dañadas)
# 0 = sin salto (video completo desde el inicio)
SALTOS_INICIO = {
    "Video01_Caqueta_A1.mp4":         0,
    "Video02_Caqueta_A2.mp4":         680,  # ← ajusta este número
    "video02_UNI_A3.mp4":             72,  # ← ajusta este número
    "Video01_Tacna_B1.mp4":           0,
    "Video03_EstadioNacional_B2.mp4": 0,
    "Video03_Aramburu_B3.mp4":        94,  # ← ajusta este número
}

# ── Cargar modelo (una sola vez para todas las cámaras) ───────────────────────
print("=" * 60)
print("  OE1 — Pipeline 6 cámaras | Metropolitano de Lima")
print("=" * 60)
print(f"\n[MODELO] Cargando {MODELO}...")
model = YOLO(MODELO)
print(f"[MODELO] Listo — clase 0 = '{model.names[0]}'")

# ── CSV consolidado (todas las cámaras en un solo archivo) ────────────────────
ts_inicio      = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_consolidado = os.path.join(CARPETA, f"dataset_OE1_CONSOLIDADO_{ts_inicio}.csv")
COLUMNAS = [
    "timestamp", "frame", "estacion", "camara",
    "personas", "confianza_promedio", "latencia_ms"
]

archivo_consolidado = open(csv_consolidado, "w", newline="", encoding="utf-8")
writer_consolidado  = csv.DictWriter(archivo_consolidado, fieldnames=COLUMNAS)
writer_consolidado.writeheader()
print(f"[CSV] Consolidado iniciado: {csv_consolidado}\n")

# ── Métricas globales (para el resumen final) ─────────────────────────────────
resumen_global = []


# ==============================================================================
# FUNCIÓN PRINCIPAL: procesar un video completo
# ==============================================================================
def procesar_camara(ruta_video, estacion, camara):
    print("\n" + "─" * 60)
    print(f"  PROCESANDO: {camara} | Estación: {estacion}")
    print(f"  Archivo   : {os.path.basename(ruta_video)}")
    print("─" * 60)

    # Verificar que el archivo existe antes de intentar abrirlo
    if not os.path.exists(ruta_video):
        print(f"  [ERROR] Archivo no encontrado: {ruta_video}")
        print(f"  [SKIP]  Saltando {camara}...\n")
        return None

    cap = cv2.VideoCapture(ruta_video)
    # Saltar la parte dañada del inicio
    segundos_skip = SALTOS_INICIO.get(os.path.basename(ruta_video), 0)
    if segundos_skip > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, segundos_skip * 1000)
        print(f"  Saltando {segundos_skip}s de intro dañada...")

    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir el video: {ruta_video}")
        return None

    # Info del video
    fps_orig     = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_seg = total_frames / fps_orig
    print(f"  Frames totales : {total_frames} (~{duracion_seg:.0f} segundos)")
    print(f"  FPS original   : {fps_orig:.1f}")

    # CSV individual para esta cámara
    nombre_csv_ind = os.path.join(
        CARPETA,
        f"dataset_OE1_{estacion}_{camara}_{ts_inicio}.csv"
    )
    archivo_ind = open(nombre_csv_ind, "w", newline="", encoding="utf-8")
    writer_ind  = csv.DictWriter(archivo_ind, fieldnames=COLUMNAS)
    writer_ind.writeheader()

    # Variables de control
    frame_num  = 0
    procesados = 0
    latencias  = []
    conteos    = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % SALTO != 0:
            continue

        # ── Inferencia ──────────────────────────────────────────────────
        t0      = time.perf_counter()
        results = model.predict(frame, verbose=False,
                                classes=CLASES, conf=CONFIANZA)
        t1      = time.perf_counter()
        lat_ms  = (t1 - t0) * 1000

        # Contar personas y confianza promedio
        personas = 0
        confs    = []
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                personas += 1
                confs.append(float(box.conf[0]))

        conf_prom = round(sum(confs) / len(confs), 3) if confs else 0.0

        latencias.append(lat_ms)
        conteos.append(personas)
        procesados += 1

        # ── Guardar en ambos CSVs ────────────────────────────────────────
        fila = {
            "timestamp":         datetime.now().isoformat(),
            "frame":             frame_num,
            "estacion":          estacion,
            "camara":            camara,
            "personas":          personas,
            "confianza_promedio": conf_prom,
            "latencia_ms":       round(lat_ms, 2),
        }
        writer_ind.writerow(fila)
        writer_consolidado.writerow(fila)

        # ── Mostrar video (opcional) ─────────────────────────────────────
        if MOSTRAR:
            annotated = results[0].plot()

            # Panel informativo sobre el frame
            cv2.rectangle(annotated, (0, 0), (380, 55), (0, 0, 0), -1)
            cv2.putText(annotated,
                        f"{estacion} | {camara} | Personas: {personas}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.putText(annotated,
                        f"Frame {frame_num}/{total_frames} | {lat_ms:.1f} ms",
                        (8, 46), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180, 255, 180), 1)

            cv2.imshow("OE1 - Metropolitano de Lima", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  [INFO] Detenido por usuario.")
                break

        # ── Log cada 100 frames procesados ──────────────────────────────
        if procesados % 100 == 0:
            prom_lat = sum(latencias[-100:]) / 100
            prom_per = sum(conteos[-100:]) / 100
            print(f"  Frame {frame_num:>5}/{total_frames} | "
                  f"Personas: {personas:>2} | "
                  f"Latencia: {lat_ms:>6.1f} ms | "
                  f"Prom-100: lat={prom_lat:.1f}ms per={prom_per:.1f}")

    # ── Cerrar recursos de esta cámara ───────────────────────────────────────
    cap.release()
    archivo_ind.close()

    # ── Resumen individual ───────────────────────────────────────────────────
    if latencias:
        resultado = {
            "camara":          camara,
            "estacion":        estacion,
            "frames_proc":     procesados,
            "lat_min":         round(min(latencias), 1),
            "lat_max":         round(max(latencias), 1),
            "lat_prom":        round(sum(latencias) / len(latencias), 1),
            "personas_prom":   round(sum(conteos) / len(conteos), 1),
            "personas_max":    max(conteos),
            "csv_individual":  nombre_csv_ind,
        }

        print(f"\n  ✓ {camara} completada")
        print(f"    Frames procesados : {procesados}")
        print(f"    Latencia promedio : {resultado['lat_prom']} ms")
        print(f"    Personas promedio : {resultado['personas_prom']}")
        print(f"    Personas máximo   : {resultado['personas_max']}")
        print(f"    CSV guardado en   : {os.path.basename(nombre_csv_ind)}")

        return resultado
    return None


# ==============================================================================
# LOOP PRINCIPAL — procesar las 6 cámaras en secuencia
# ==============================================================================
print(f"\nIniciando procesamiento de {len(CAMARAS)} cámaras...\n")

for nombre_archivo, estacion, camara in CAMARAS:
    ruta_completa = os.path.join(CARPETA, nombre_archivo)
    resultado = procesar_camara(ruta_completa, estacion, camara)
    if resultado:
        resumen_global.append(resultado)

# Cerrar recursos globales
cv2.destroyAllWindows()
archivo_consolidado.close()

# ==============================================================================
# RESUMEN FINAL — métricas para tu tabla de resultados de tesis
# ==============================================================================
print("\n" + "=" * 60)
print("  RESUMEN FINAL OE1 — TABLA DE RESULTADOS")
print("=" * 60)
print(f"  {'Cámara':<12} {'Estación':<10} {'Frames':>7} "
      f"{'Lat.Prom':>9} {'Lat.Min':>8} {'Lat.Max':>8} "
      f"{'Per.Prom':>9} {'Per.Max':>8}")
print("  " + "-" * 58)

lat_todas = []
for r in resumen_global:
    print(f"  {r['camara']:<12} {r['estacion']:<10} {r['frames_proc']:>7} "
          f"{r['lat_prom']:>8.1f}ms {r['lat_min']:>7.1f}ms "
          f"{r['lat_max']:>7.1f}ms {r['personas_prom']:>8.1f} "
          f"{r['personas_max']:>8}")
    lat_todas.append(r['lat_prom'])

if lat_todas:
    print("  " + "-" * 58)
    print(f"\n  Latencia promedio global : {sum(lat_todas)/len(lat_todas):.1f} ms")
    print(f"  Cámaras procesadas       : {len(resumen_global)} / {len(CAMARAS)}")
    print(f"\n  CSV consolidado          : {os.path.basename(csv_consolidado)}")

print("\n" + "=" * 60)
print("  Dimensión 1 — Cobertura")
print(f"  Estaciones cubiertas : 2 / 2  (Caqueta, Tacna)")
print(f"  Cámaras operativas   : {len(resumen_global)} / {len(CAMARAS)}")
print("─" * 60)
print("  Dimensión 2 — Latencia")
if lat_todas:
    print(f"  Latencia inferencia  : {sum(lat_todas)/len(lat_todas):.1f} ms promedio global")
    print(f"  Meta establecida     : ≤ 200 ms  →  {'✓ CUMPLIDA' if sum(lat_todas)/len(lat_todas) <= 200 else '✗ NO CUMPLIDA'}")
print("=" * 60)
print("\n  Pipeline OE1 completado.")
print("  Próximo paso → Fase 3: montar red GNS3 y streams RTSP\n")