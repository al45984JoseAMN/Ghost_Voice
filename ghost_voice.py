import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

# ==========================================
# 1. CARGA Y ANÁLISIS DEL AUDIO
# ==========================================
audio_path = "amorphous.mp3"

#Lista de Canciones disponibles
#Ghost_Voices
#Mirage
#Butterflies
#Something_Comforting
#amorphous
#Shelter
#Mumbai_Power
#Blame

print(
    "Cargando la canción y configurando el ecualizador en tiempo real..."
)
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

S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
S_db = librosa.power_to_db(S, ref=np.max)
bound_frames = librosa.segment.agglomerative(S_db, k=min(6, S_db.shape[1] // 50))
bound_times = librosa.frames_to_time(
    bound_frames, sr=sr, hop_length=hop_length
)
bound_times = np.concatenate(([0], bound_times, [times[-1]]))
bound_times = np.unique(np.sort(bound_times))

# ==========================================
# 4. CONFIGURACIÓN ESTÉTICA (DARK MODE NEON + LIVE SPECTRUM)
# ==========================================
plt.style.use("dark_background")
plt.ion()

fig, axes = plt.subplots(6, 1, figsize=(15, 14), facecolor="#0e1117")
ax1, ax2, ax3, ax4, ax5, ax6 = axes

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
  ax.set_facecolor("#161b22")
  for j in range(len(bound_times) - 1):
    ax.axvspan(
        bound_times[j],
        bound_times[j + 1],
        color=colores_secciones[j % len(colores_secciones)],
        alpha=0.4,
    )
  ax.plot(
      times, energia, color=color_linea, alpha=0.9, linewidth=1.2, label=titulo
  )
  ax.fill_between(times, energia, color=color_relleno, alpha=0.25)
  ax.set_title(titulo, fontsize=9, fontweight="bold", color="#f0f6fc", loc="left")
  ax.set_ylabel("Energía", fontsize=8, color="#8b949e")
  ax.grid(True, linestyle=":", alpha=0.2, color="#30363d")

  for t_drop in tiempos_drops:
    ax.axvline(
        x=t_drop, color="#ff4757", linestyle="--", linewidth=1.2, alpha=0.6
    )

  v_line = ax.axvline(
      x=0, color="#00d2d3", linestyle="-", linewidth=1.8, alpha=0.9
  )
  v_lines.append(v_line)

  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_color("#30363d")
  ax.spines["bottom"].set_color("#30363d")

ax5.set_xlabel("Tiempo transcurrido general (segundos)", fontsize=9, color="#f0f6fc")

# Configuración de la 6ª Gráfica: Espectro Instantáneo en Tiempo Real
ax6.set_facecolor("#161b22")
(live_spectrum_line,) = ax6.plot(
    freqs, magnitude[:, 0], color="#00d2d3", linewidth=1.2
)
ax6.fill_between(freqs, magnitude[:, 0], color="#00d2d3", alpha=0.2)
ax6.set_title(
    "⚡ ESPECTRO DE FRECUENCIA EN TIEMPO REAL (Ecualizador vivo)",
    fontsize=9,
    fontweight="bold",
    color="#00d2d3",
    loc="left",
)
ax6.set_xlabel("Frecuencia (Hz)", fontsize=8, color="#8b949e")
ax6.set_ylabel("Intensidad", fontsize=8, color="#8b949e")
ax6.set_xlim(0, 8000)
ax6.set_ylim(0, np.max(magnitude) * 0.4)
ax6.grid(True, linestyle=":", alpha=0.2, color="#30363d")
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)
ax6.spines["left"].set_color("#30363d")
ax6.spines["bottom"].set_color("#30363d")

plt.tight_layout()

# ==========================================
# 5. REPRODUCCIÓN Y ANIMACIÓN EN VIVO
# ==========================================
print("\n--- PROGRAMA LISTO CON ESPECTRO EN VIVO ---")
print(f"• Tempo estimado (BPM): {tempo:.2f}")
print("• Reproduciendo música y actualizando gráfico en tiempo real...")
print("-------------------------------------------\n")

sd.play(y, sr)
start_time = time.time()
duracion_total = len(y) / sr

try:
  while True:
    current_time = time.time() - start_time
    if current_time >= duracion_total:
      break

    # 1. Mover las líneas de tiempo en las 5 gráficas superiores
    for v_line in v_lines:
      v_line.set_xdata([current_time, current_time])

    # 2. Actualizar el espectro instantáneo en la 6ª gráfica
    current_frame = librosa.time_to_frames(
        current_time, sr=sr, hop_length=hop_length
    )
    current_frame = min(current_frame, magnitude.shape[1] - 1)

    current_magnitude = magnitude[:, current_frame]
    live_spectrum_line.set_ydata(current_magnitude)

    # Limpiar las colecciones anteriores de forma compatible y redibujar el relleno
    for coll in ax6.collections:
      coll.remove()
    ax6.fill_between(freqs, current_magnitude, color="#00d2d3", alpha=0.2)

    plt.draw()
    plt.pause(0.02)

except KeyboardInterrupt:
  sd.stop()

sd.wait()
plt.ioff()
plt.show()