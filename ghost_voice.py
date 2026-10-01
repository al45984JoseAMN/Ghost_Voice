import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

# ==========================================
# 1. CARGA Y ANÁLISIS DEL AUDIO
# ==========================================
audio_path = "Ghost_Voices.mp3"

print("Cargando la canción y aplicando diseño estético avanzado...")
y, sr = librosa.load(audio_path)

# Calcular BPM
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
if isinstance(tempo, np.ndarray):
  tempo = tempo[0]

hop_length = 512
n_fft = 2048
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
magnitude = np.abs(D)
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

# ==========================================
# 2. DEFINICIÓN DE BANDAS Y CÁLCULO DE ENERGÍA
# ==========================================
idx_subgraves = np.where((freqs >= 20) & (freqs < 60))[0]
idx_graves = np.where((freqs >= 60) & (freqs < 250))[0]
idx_medios = np.where((freqs >= 250) & (freqs < 2000))[0]
idx_agudos = np.where((freqs >= 2000) & (freqs < 6000))[0]
idx_brillos = np.where((freqs >= 6000) & (freqs <= sr / 2))[0]

energia_subgraves = np.sum(magnitude[idx_subgraves, :] ** 2, axis=0)
energia_graves = np.sum(magnitude[idx_graves, :] ** 2, axis=0)
energia_medios = np.sum(magnitude[idx_medios, :] ** 2, axis=0)
energia_agudos = np.sum(magnitude[idx_agudos, :] ** 2, axis=0)
energia_brillos = np.sum(magnitude[idx_brillos, :] ** 2, axis=0)

times = librosa.frames_to_time(
    np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length
)

# ==========================================
# 3. DETECCIÓN DE DROPS Y ESTRUCTURA
# ==========================================
energia_bajos_total = energia_subgraves + energia_graves
derivada_energia = np.gradient(energia_bajos_total)
umbral_drop = np.mean(derivada_energia) + 2.5 * np.std(derivada_energia)
picos_drops_idx = np.where(derivada_energia > umbral_drop)[0]

tiempos_drops = []
ultimo_tiempo = -10
for idx in picos_drops_idx:
  t = times[idx]
  if t - ultimo_tiempo > 8:
    tiempos_drops.append(t)
    ultimo_tiempo = t

# Segmentación estructural
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
S_db = librosa.power_to_db(S, ref=np.max)
bound_frames = librosa.segment.agglomerative(S_db, k=min(6, S_db.shape[1] // 50))
bound_times = librosa.frames_to_time(
    bound_frames, sr=sr, hop_length=hop_length
)
bound_times = np.concatenate(([0], bound_times, [times[-1]]))
bound_times = np.unique(np.sort(bound_times))

secciones_etiquetas = []
for i in range(len(bound_times) - 1):
  t_ini, t_fin = bound_times[i], bound_times[i + 1]
  idx_sec = np.where((times >= t_ini) & (times <= t_fin))[0]
  energia_prom = (
      np.mean(energia_bajos_total[idx_sec]) if len(idx_sec) > 0 else 0
  )
  if i == 0:
    etiqueta = "Intro"
  elif i == len(bound_times) - 2:
    etiqueta = "Outro"
  elif energia_prom > np.mean(energia_bajos_total) * 1.2:
    etiqueta = "Drop / Clímax"
  else:
    etiqueta = "Breakdown"
  secciones_etiquetas.append(etiqueta)

# ==========================================
# 4. CONFIGURACIÓN ESTÉTICA (DARK MODE NEON)
# ==========================================
plt.style.use(
    "dark_background"
)  # Activa el estilo oscuro nativo de Matplotlib
plt.ion()

fig, axes = plt.subplots(5, 1, figsize=(15, 11), facecolor="#0e1117")
ax1, ax2, ax3, ax4, ax5 = axes

# Colores estilo Neón y paleta cyberpunk
bandas_configs = [
    (ax1, energia_subgraves, "SUB-GRAVES (20 - 60 Hz)", "#ff3838", "#ff4d4d"),
    (ax2, energia_graves, "GRAVES - KICK & BASS (60 - 250 Hz)", "#ff9f43", "#ffb142"),
    (
        ax3,
        energia_medios,
        "MEDIOS - VOCALES Y MELODÍAS (250 - 2k Hz)",
        "#2ed573",
        "#26af5f",
    ),
    (ax4, energia_agudos, "AGUDOS - ARMÓNICOS (2k - 6k Hz)", "#1e90ff", "#0080ff"),
    (
        ax5,
        energia_brillos,
        "BRILLOS - HI-HATS (6k - 20k Hz)",
        "#9b59b6",
        "#8e44ad",
    ),
]

colores_secciones = ["#1f2937", "#111827", "#1f2937", "#374151", "#111827"]
v_lines = []

for ax, energia, titulo, color_linea, color_relleno in bandas_configs:
  # Fondo oscuro personalizado para cada subplot
  ax.set_facecolor("#161b22")

  # Dibujar bloques estructurales sutiles en el fondo
  for j in range(len(bound_times) - 1):
    ax.axvspan(
        bound_times[j],
        bound_times[j + 1],
        color=colores_secciones[j % len(colores_secciones)],
        alpha=0.4,
    )

  # Curva principal con neón y área rellenada debajo
  ax.plot(
      times, energia, color=color_linea, alpha=0.9, linewidth=1.2, label=titulo
  )
  ax.fill_between(
      times, energia, color=color_relleno, alpha=0.25
  )  # Efecto de volumen

  ax.set_title(titulo, fontsize=9, fontweight="bold", color="#f0f6fc", loc="left")
  ax.set_ylabel("Energía", fontsize=8, color="#8b949e")
  ax.grid(True, linestyle=":", alpha=0.2, color="#30363d")

  # Marcar Drops con líneas rojas traslúcidas
  for t_drop in tiempos_drops:
    ax.axvline(
        x=t_drop, color="#ff4757", linestyle="--", linewidth=1.2, alpha=0.6
    )

  # Cabezal de reproducción (Línea Cían Neón)
  v_line = ax.axvline(
      x=0, color="#00d2d3", linestyle="-", linewidth=1.8, alpha=0.9
  )
  v_lines.append(v_line)

  # Limpiar bordes estéticos (quitar bordes superiores y derechos)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_color("#30363d")
  ax.spines["bottom"].set_color("#30363d")

ax5.set_xlabel("Tiempo transcurrido (segundos)", fontsize=10, color="#f0f6fc")
plt.tight_layout()

# ==========================================
# 5. REPRODUCCIÓN Y SINCRONIZACIÓN FLUIDA
# ==========================================
print("\n--- ANÁLISIS ESTÉTICO LISTO ---")
print(f"• Tempo estimado (BPM): {tempo:.2f}")
print("• Interfaz gráfica configurada en Modo Oscuro / Neón.")
print("--------------------------------\n")

print("¡Reproduciendo audio y sincronizando visualizador estético!")
sd.play(y, sr)

start_time = time.time()
duracion_total = len(y) / sr

try:
  while True:
    current_time = time.time() - start_time
    if current_time >= duracion_total:
      break

    for v_line in v_lines:
      v_line.set_xdata([current_time, current_time])

    plt.draw()
    plt.pause(0.03)

except KeyboardInterrupt:
  sd.stop()

sd.wait()
plt.ioff()
plt.show()