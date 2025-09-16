import cv2
import os
import hashlib
import time

# === Detector de rostros ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

# === Cargar imágenes y nombres de Pokémons ===
pokemons_dir = "Pokemon"
pokemon_imgs = []
pokemon_nombres = []

for fname in sorted(os.listdir(pokemons_dir)):
    if fname.endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(pokemons_dir, fname)
        img = cv2.imread(path)  # BGR sin alfa
        pokemon_imgs.append(img)
        nombre = os.path.splitext(fname)[0]
        pokemon_nombres.append(nombre)

# === Variables globales ===
frame_final = None
foto_tomada = False
inicio = None
DURACION = 5  # segundos de cuenta regresiva
boton_rect = (30, 30, 220, 70)  # x, y, ancho, alto
pokemon_actuales = {}

# === Funciones ===
def overlay_image(bg, fg, x, y, w, h):
    fg = cv2.resize(fg, (w, h))
    bg[y:y+h, x:x+w] = fg
    return bg

def pokemon_for_face(rect):
    x,y,w,h = rect
    hsh = hashlib.sha256(f"{x}{y}{w}{h}".encode()).hexdigest()
    idx = int(hsh, 16) % len(pokemon_imgs)
    return idx, pokemon_imgs[idx], pokemon_nombres[idx]

def dibujar_boton(frame):
    x, y, w, h = boton_rect
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), -1)
    cv2.putText(frame, "Reiniciar", (x+15, y+50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

def click_event(event, x, y, flags, param):
    global inicio, foto_tomada, frame_final, pokemon_actuales
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = boton_rect
        if bx <= x <= bx+bw and by <= y <= by+bh:
            # Reiniciar todo
            inicio = None
            foto_tomada = False
            frame_final = None
            pokemon_actuales = {}

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

    # Tomamos hasta 4 personas
    faces = faces[:4]

    # Asignar Pokémon a cada cara
    for i, rect in enumerate(faces):
        if i not in pokemon_actuales or not foto_tomada:
            idx, img, nombre = pokemon_for_face(rect)
            pokemon_actuales[i] = img
        x, y, w, h = rect
        px, py = x, y - h
        if px >= 0 and py >= 0 and px+w <= frame.shape[1] and py+h <= frame.shape[0]:
            frame = overlay_image(frame, pokemon_actuales[i], px, py, w, h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Inicializar tiempo si hay caras
    if len(faces) > 0 and not inicio:
        inicio = time.time()

    # Congelar imagen después de DURACION segundos
    if inicio and not foto_tomada and (time.time() - inicio >= DURACION):
        frame_final = frame.copy()
        foto_tomada = True

    if foto_tomada and frame_final is not None:
        frame = frame_final.copy()

    # Texto en la parte superior sobre los Pokémon
    if len(faces) > 0:
        texto = "¿Que Pokemon eres?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        escala = 1.5
        grosor = 3
        (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, font, escala, grosor)
        x_texto = (frame.shape[1] - ancho_texto) // 2
        y_texto = 50  # cerca del borde superior
        cv2.putText(frame, texto, (x_texto, y_texto), font, escala, (0,255,255), grosor)

    # Dibujar botón
    dibujar_boton(frame)

    cv2.imshow("Pokémon filtro estilo TikTok", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()