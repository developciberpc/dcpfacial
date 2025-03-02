import cv2
import dlib

# Cargar el detector frontal de dlib
detector = dlib.get_frontal_face_detector()

# Cargar el predictor de forma (modifica la ruta si es necesario)
predictor = dlib.shape_predictor("C:/Users/jalex/Desktop/reconosimiento-FACIAL/shape_predictor_68_face_landmarks.dat")

# Iniciar captura de video
captura = cv2.VideoCapture(0)  # Asegúrate de que el índice de la cámara sea correcto (0 o 1)

if not captura.isOpened():
    print("No se puede abrir la cámara")
    exit()

while True:
    ret, frame = captura.read()

    if not ret:
        print("Error al capturar el frame")
        break

    # Mostrar el frame capturado antes de cualquier procesamiento
    cv2.imshow("Video en directo", frame)

    # Probar detectar caras en la imagen en color (BGR)
    try:
        faces = detector(frame)  # Pasar la imagen en color
        for face in faces:
            x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        # Mostrar la imagen con los rectángulos de detección
        cv2.imshow("Reconocimiento facial", frame)
    
    except Exception as e:
        print(f"Error al detectar caras en color: {e}")

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Asegurarse de que la imagen está en formato de 8-bit
    gray = cv2.convertScaleAbs(gray)

    # Verificar el formato de la imagen después de la conversión
    print(f"Formato de imagen: {gray.dtype}, Dimensiones: {gray.shape}")

    try:
        # Detectar caras en la imagen en escala de grises
        faces = detector(gray)

        # Recorrer cada cara detectada y dibujar un rectángulo
        for face in faces:
            x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        # Mostrar la imagen con los rectángulos de detección
        cv2.imshow("Reconocimiento facial", frame)

    except Exception as e:
        print(f"Error al detectar caras en escala de grises: {e}")

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la captura y cerrar ventanas
captura.release()
cv2.destroyAllWindows()
