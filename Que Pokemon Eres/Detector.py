import cv2
import os
import hashlib
import time
import numpy as np
import random

# === Detector de rostros ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

# === Cargar imágenes y nombres de Pokémons ===
pokemons_dir = "Pokemon"
pokemon_imgs = []
pokemon_nombres = [],

for fname in sorted(os.listdir(pokemons_dir)):
    if fname.endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(pokemons_dir, fname)
        img = cv2.imread(path)
        pokemon_imgs.append(img)
        nombre = os.path.splitext(fname)[0]
        pokemon_nombres.append(nombre)

# === Variables globales ===
frame_final = None
foto_tomada = False
inicio = None
DURACION = 5  # segundos de cuenta regresiva
boton_rect = (30, 30, 120, 40)
boton_save_rect = (170, 30, 120, 40)
pokemon_actuales = {}
confeti_coords = []
confeti_anim = 0
fadein_frames = 15

# === Funciones ===
def overlay_image_fade(bg, fg, x, y, w, h, alpha=1.0):
    fg = cv2.resize(fg, (w, h))
    if fg.shape[2] == 3:
        fg = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)
    roi = bg[y:y+h, x:x+w]
    if roi.shape[2] == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
    blended = cv2.addWeighted(roi, 1-alpha, fg, alpha, 0)
    bg[y:y+h, x:x+w] = blended[:, :, :3]
    return bg

def pokemon_for_face(rect):
    x,y,w,h = rect
    hsh = hashlib.sha256(f"{x}{y}{w}{h}".encode()).hexdigest()
    idx = int(hsh, 16) % len(pokemon_imgs)
    return idx, pokemon_imgs[idx], pokemon_nombres[idx]

def dibujar_boton(frame):
    x, y, w, h = boton_rect
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), -1)
    cv2.putText(frame, "Reiniciar", (x+10, y+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

def dibujar_boton_save(frame):
    x, y, w, h = boton_save_rect
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,128,0), -1)
    cv2.putText(frame, "Guardar", (x+15, y+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

def click_event(event, x, y, flags, param):
    global inicio, foto_tomada, frame_final, pokemon_actuales, confeti_coords, confeti_anim
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = boton_rect
        bx2, by2, bw2, bh2 = boton_save_rect
        if bx <= x <= bx+bw and by <= y <= by+bh:
            # Reiniciar todo
            inicio = None
            foto_tomada = False
            frame_final = None
            pokemon_actuales = {}
            confeti_coords = []
            confeti_anim = 0
        elif foto_tomada and bx2 <= x <= bx2+bw2 and by2 <= y <= by2+bh2:
            # Guardar imagen
            if frame_final is not None:
                nombre_archivo = f"foto_pokemon_{int(time.time())}.png"
                cv2.imwrite(nombre_archivo, frame_final)
                print(f"Foto guardada como {nombre_archivo}")

def draw_countdown(frame, segundos_restantes):
    texto = f"{segundos_restantes}"
    font = cv2.FONT_HERSHEY_DUPLEX
    escala = 4
    grosor = 8
    colores = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
    color = colores[segundos_restantes % len(colores)]
    (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, font, escala, grosor)
    x = (frame.shape[1] - ancho_texto) // 2
    y = (frame.shape[0] + alto_texto) // 2
    cv2.putText(frame, texto, (x, y), font, escala, color, grosor, cv2.LINE_AA)

def draw_pokemon_name(frame, nombre, px, py, w):
    (text_w, text_h), _ = cv2.getTextSize(nombre, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py-40), (px+text_w+10, py-10), (0,0,0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, nombre, (px+5, py-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

def generar_confeti(frame, cantidad=150):
    coords = []
    for _ in range(cantidad):
        x = random.randint(0, frame.shape[1]-1)
        y = random.randint(0, frame.shape[0]-1)
        color = tuple([random.randint(0,255) for _ in range(3)])
        radius = random.randint(3, 7)
        coords.append([x, y, color, radius, random.randint(2,6)])
    return coords

def animar_confeti(coords, frame_shape):
    for c in coords:
        c[1] += c[4]  # velocidad vertical
        if c[1] > frame_shape[0]:
            c[1] = random.randint(-20, 0)
            c[0] = random.randint(0, frame_shape[1]-1)

def dibujar_confeti(frame, coords):
    for x, y, color, radius, _ in coords:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)

# === Ventana y callback ===
cv2.namedWindow("Pokémon filtro estilo TikTok", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Pokémon filtro estilo TikTok", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Pokémon filtro estilo TikTok", click_event)

# === Webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = faces[:4]

    # Animación de aparición (fade-in)
    fade_alpha = 1.0
    if inicio and not foto_tomada:
        frames_passed = int((time.time() - inicio) * 30)
        fade_alpha = min(1.0, frames_passed / fadein_frames)

    # Asignar Pokémon a cada cara
    for i, rect in enumerate(faces):
        if i not in pokemon_actuales or not foto_tomada:
            idx, img, nombre = pokemon_for_face(rect)
            pokemon_actuales[i] = (img, nombre)
        x, y, w, h = rect
        px, py = x, y - h
        img, nombre = pokemon_actuales[i]
        if px >= 0 and py >= 0 and px+w <= frame.shape[1] and py+h <= frame.shape[0]:
            if not foto_tomada:
                frame = overlay_image_fade(frame, img, px, py, w, h, alpha=fade_alpha)
            else:
                frame = overlay_image_fade(frame, img, px, py, w, h, alpha=1.0)
            draw_pokemon_name(frame, nombre, px, py, w)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Inicializar tiempo si hay caras
    if len(faces) > 0 and not inicio:
        inicio = time.time()

    # Cuenta regresiva visual
    if inicio and not foto_tomada:
        segundos_restantes = max(0, int(DURACION - (time.time() - inicio)))
        draw_countdown(frame, segundos_restantes)

    # Congelar imagen después de DURACION segundos
    if inicio and not foto_tomada and (time.time() - inicio >= DURACION):
        frame_final = frame.copy()
        foto_tomada = True
        confeti_coords = generar_confeti(frame_final, cantidad=150)
        confeti_anim = 0

    # Animación de confeti
    if foto_tomada and frame_final is not None:
        frame = frame_final.copy()
        if confeti_anim < 60:  # 2 segundos de animación
            animar_confeti(confeti_coords, frame.shape)
            dibujar_confeti(frame, confeti_coords)
            confeti_anim += 1

    # Texto en la parte superior sobre los Pokémon
    if len(faces) > 0:
        texto = "¿Qué Pokémon eres?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        escala = 1.5
        grosor = 3
        (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, font, escala, grosor)
        x_texto = (frame.shape[1] - ancho_texto) // 2
        y_texto = 50
        cv2.putText(frame, texto, (x_texto, y_texto), font, escala, (0,255,255), grosor)

    dibujar_boton(frame)
    if foto_tomada:
        dibujar_boton_save(frame)

    cv2.imshow("Pokémon filtro estilo TikTok", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()