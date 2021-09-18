import cv2

detector_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read('classificadorEigen.yml')
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while True:
    conectado, imagem = camera.read()
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces_detectadas = detector_face.detectMultiScale(imagem_cinza,
                                                      scaleFactor=1.5,
                                                      minSize=(30, 30))

    for (x, y, l, a) in faces_detectadas:
        imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagem_face)
        cv2.putText(imagem, str(id), (x, y + (a + 30)), font, 2, (0, 0, 255))

    cv2.imshow('Face', imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
