import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create(num_components=10)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]  # Diretorio onde as fotos serao salvas
    faces = []  # Lista de faces3
    ids = []  # Lista de identificadores
    for caminho_imagem in caminhos:
        imagem_face = cv2.cvtColor(cv2.imread(caminho_imagem), cv2.COLOR_BGR2GRAY)  # Fazendo a leitura das imagens
        id = int(os.path.split(caminho_imagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagem_face)
    return np.array(ids), faces


ids, faces = getImagemComId()

print('Treinando algoritmo...')
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')  # salvando arquivos para o eigen_face

fisherface.train(faces, ids)
fisherface.write('classificadorEigen.yml')  # salvando arquivos para o fisher_face
