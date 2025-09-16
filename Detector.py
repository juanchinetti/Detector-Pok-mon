import cv2
import os
import hashlib
import time
import random

# === Detector de rostros ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

# === Cargar imágenes de Pokémons ===
pokemons_dir = "Pokemon"
pokemon_imgs = []
pokemon_nombres = []

for fname in sorted(os.listdir(pokemons_dir)):
    if fname.endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(pokemons_dir, fname)
        img = cv2.imread(path)
        pokemon_imgs.append(img)
        pokemon_nombres.append(os.path.splitext(fname)[0])

# === Variables globales ===
frame_final = None
foto_tomada = False
inicio = None
DURACION = 5  # segundos antes de congelar
boton_rect = (20, 650, 150, 50)  # esquina inferior izquierda

# === Funciones ===
def overlay_image(bg, fg, x, y, w, h):
    fg = cv2.resize(fg, (w, h))
    bg[y:y+h, x:x+w] = fg
    return bg

def elegir_pokemon():
    idx = random.randint(0, len(pokemon_imgs) - 1)
    return pokemon_imgs[idx], pokemon_nombres[idx]

def dibujar_boton(frame):
    x, y, w, h = boton_rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), -1)
    cv2.putText(frame, "Reiniciar", (x+10, y+35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def click_event(event, x, y, flags, param):
    global inicio, foto_tomada, frame_final
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = boton_rect
        if bx <= x <= bx+bw and by <= y <= by+bh:
            inicio = None
            foto_tomada = False
            frame_final = None

# === Ventana y callback ===
cv2.namedWindow("Pokemon filtro estilo TikTok", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Pokemon filtro estilo TikTok", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Pokemon filtro estilo TikTok", click_event)

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

    pokemon_actual = None
    nombre_actual = None

    # Solo consideramos la primera cara detectada
    if len(faces) > 0:
        x, y, w, h = faces[0]

        # Durante los primeros DURACION segundos, Pokémon cambia
        if not inicio:
            inicio = time.time()

        if not foto_tomada:
            pokemon_actual, nombre_actual = elegir_pokemon()
        else:
            # Congelado, mantener último Pokémon
            pokemon_actual, nombre_actual = pokemon_actual, nombre_actual

        px, py = x, y - h
        if px >= 0 and py >= 0 and px+w <= frame.shape[1] and py+h <= frame.shape[0]:
            frame = overlay_image(frame, pokemon_actual, px, py, w, h)

    # Congelar después de DURACION segundos
    if inicio and not foto_tomada and (time.time() - inicio >= DURACION):
        frame_final = frame.copy()
        foto_tomada = True

    if foto_tomada and frame_final is not None:
        frame = frame_final.copy()

        if nombre_actual:
            texto = f"Tu Pokemon es {nombre_actual}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            escala = 1.5
            grosor = 3
            (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, font, escala, grosor)
            x_texto = (frame.shape[1] - ancho_texto) // 2
            y_texto = frame.shape[0] - 50
            cv2.putText(frame, texto, (x_texto, y_texto), font, escala, (0,255,255), grosor)

    dibujar_boton(frame)
    cv2.imshow("Pokémon filtro estilo TikTok", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
