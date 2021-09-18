import cv2
import numpy as np

classificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classificador_olho = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
camera = cv2.VideoCapture(0)  # Indicando que a camera usada vai ser a webcam (notebook)
amostra = 1  # Controle da quantidade de fotos tiradas
numero_amostras = 25  # Quantidade de amostras que serao coletadas
identity = input('Digite seu identificador:')  # Recebendo identificacao da pessoa
largura, altura = 220, 220  # Controle do tamanho da foto tirada
print('Capturando as faces')

while True:
    conectado, imagem = camera.read()  # Leitura da webcam
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)  # Convertendo imagem capturada para escala de cinza
    faces_detectadas = classificador.detectMultiScale(imagem_cinza,  # Detectando faces na imagem em esc de cinza
                                                      scaleFactor=1.5,
                                                      minSize=(150, 150))
    # Desenhando um retangulo no entorno do rosto
    for (x, y, l, a) in faces_detectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)  # Definindo cor do quadrado ao redor da face
        regiao = imagem[y:y + a, x:x + l]  # Definindo regiao de captura dos olhos
        regiao_cinza_olho = cv2.cvtColor(regiao, cv2.COLOR_RGB2GRAY)  # Convertendo imagem dos olhos para esc de cinza
        olhos_detectados = classificador_olho.detectMultiScale(regiao_cinza_olho)  # Detectando os olhos em img
        # na escala de cinza

        for (ox, oy, ol, oa) in olhos_detectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
            # detectando botao 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagem_cinza) > 110:
                    imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l])
                    cv2.imwrite('fotos/pessoa.' + str(identity) + '.' + str(amostra) + 'jpg', imagem_face)
                    print('[foto ' + str(amostra) + 'capturada com sucesso]')
                    amostra += 1
    cv2.imshow('Face', imagem)  # mostrando imagem capturada pela webcam
    cv2.waitKey(1)
    if amostra >= numero_amostras + 1:
        break

print('Faces capturadas com sucesso')
camera.release()  # Abrindo a camera
cv2.destroyAllWindows()  # Liberando memoria
