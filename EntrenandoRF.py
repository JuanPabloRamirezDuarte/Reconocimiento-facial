import cv2
import os
import numpy as np

datapath = 'C:/Users/Juan Pablo/Desktop/prueba/Data'
peopleList = os.listdir(datapath)
print('Lista de personas:',peopleList)

labels = []
facesData = []
label = 0 

for nameDir in peopleList:
	personPath = datapath + '/' + nameDir
	print('Leyendo imagenes')

	for fileName in os.listdir(personPath):
		print('Rostros', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath + '/' + fileName,0))
		image = cv2.imread(personPath+'/'+fileName,0)
		#cv2.imshow('image', image)
		#cv2.waitKey(10)
	label = label + 1
#print('labels=', labels)
#print('numero de etiquetas 0:',np.count_nonzero(np.array(labels)==0))
#print('numero de etiquetas 0:',np.count_nonzero(np.array(labels)==1))

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print('entrenando')
face_recognizer.train(facesData, np.array(labels))


#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloLBPHFace.xml')

print('modelo almacenado')