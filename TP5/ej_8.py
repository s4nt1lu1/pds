import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import resample_poly, butter, filtfilt
import sys
import os

# --- 1. Definición estricta de filtros ---
def butter_bandpass(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def butter_lowpass(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data)

# --- 2. Parámetros del Sistema (Ejercicio 8) ---
fs_ecg = 250
fs_pa = 100
fs_voz = 8000
fs_comun = 150000 
f1, f2, f3 = 30000, 40000, 50000

# Ventana temporal total a pre-calcular (1 segundo)
duracion = 1.0 

# --- 3. Validación y carga directa de los archivos físicos ---
archivos_requeridos = ['ecg.txt', 'pa.txt', 'voz.txt']
for archivo in archivos_requeridos:
    if not os.path.exists(archivo):
        print(f"Error crítico: Archivo '{archivo}' no encontrado en el directorio actual.")
        sys.exit(1)

# Ingesta
x1_raw = np.loadtxt('ecg.txt')[:int(duracion * fs_ecg)]
x2_raw = np.loadtxt('pa.txt')[:int(duracion * fs_pa)]
x3_raw = np.loadtxt('voz.txt')[:int(duracion * fs_voz)]

# Sobremuestreo al ratio del transmisor común
x1 = resample_poly(x1_raw, fs_comun, fs_ecg)
x2 = resample_poly(x2_raw, fs_comun, fs_pa)
x3 = resample_poly(x3_raw, fs_comun, fs_voz)

# Alineación de vectores
min_len = min(len(x1), len(x2), len(x3))
x1, x2, x3 = x1[:min_len], x2[:min_len], x3[:min_len]
t = np.arange(min_len) / fs_comun

# Multiplexación FDM (Transmisor)
senal_tx = (x1 * np.sin(2 * np.pi * f1 * t) + 
            x2 * np.sin(2 * np.pi * f2 * t) + 
            x3 * np.sin(2 * np.pi * f3 * t))

# Demodulación analítica (Receptor - Canal 1 ECG)
bw_ecg = fs_ecg / 2.0  # Límite estricto de Nyquist (125 Hz)
r1 = butter_bandpass(senal_tx, f1 - bw_ecg, f1 + bw_ecg, fs_comun)
ecg_rec = butter_lowpass(r1 * np.sin(2 * np.pi * f1 * t), bw_ecg, fs_comun) * 2.0

# --- 4. Motor de Renderizado Nativo ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
fig.canvas.manager.set_window_title('Simulación FDM - Transmisión y Recepción')
fig.tight_layout(pad=3.0)

line_x1, = ax1.plot([], [], 'b', lw=2)
ax1.set_xlim(0, 0.05) # Ventana de observación fija de 50 milisegundos
ax1.set_ylim(-np.max(np.abs(x1))*1.2, np.max(np.abs(x1))*1.2)
ax1.set_title('Transmisor: Canal 1 (Señal real de ECG)')
ax1.grid(True)

line_tx, = ax2.plot([], [], 'k', lw=1)
ax2.set_xlim(0, 0.05)
ax2.set_ylim(-np.max(np.abs(senal_tx))*1.2, np.max(np.abs(senal_tx))*1.2)
ax2.set_title('Enlace: Modulación FDM Completa')
ax2.grid(True)

line_rec, = ax3.plot([], [], 'r--', lw=2)
ax3.set_xlim(0, 0.05)
ax3.set_ylim(-np.max(np.abs(ecg_rec))*1.2, np.max(np.abs(ecg_rec))*1.2)
ax3.set_title('Receptor: ECG Demodulado a Banda Base')
ax3.grid(True)

# Parámetros del motor de animación
window_size = int(fs_comun * 0.05) # 50 ms en muestras
paso = int(fs_comun * 0.005)       # Avance de 5 ms por frame

def init():
    line_x1.set_data([], [])
    line_tx.set_data([], [])
    line_rec.set_data([], [])
    return line_x1, line_tx, line_rec

def update(frame):
    idx_start = frame * paso
    idx_end = idx_start + window_size
    
    # Detención de la ventana al alcanzar el final del buffer precalculado
    if idx_end > len(t):
        idx_start = len(t) - window_size
        idx_end = len(t)
        
    t_window = t[0:window_size] 
    
    # Asignación de bloques de memoria precalculados al gráfico
    line_x1.set_data(t_window, x1[idx_start:idx_end])
    line_tx.set_data(t_window, senal_tx[idx_start:idx_end])
    line_rec.set_data(t_window, ecg_rec[idx_start:idx_end])
    
    return line_x1, line_tx, line_rec

frames_totales = (len(t) - window_size) // paso

# La referencia a 'ani' es obligatoria para evitar que el recolector de basura elimine la animación
ani = animation.FuncAnimation(fig, update, frames=frames_totales, init_func=init, blit=True, interval=30, repeat=True)

# Lanza la ventana gráfica interactiva
plt.show()