import numpy as np
import codigo.funcionesAux.visualize as visualize
from munkres import Munkres, print_matrix

# paquetes para el tracker
import imutils
import time
import cv2

def iniciarTrackers(objetos,frame):
    for i in range(len(objetos['ids'])):
        objetos['age'][i]+=1
        #hay que empezar a trackearlo porque lleva un frame sin aparecer
        if objetos['age'][i]==2:
            #seleccionar la bounding box que se quiere seguir
            y1, x1, y2, x2 = objetos['rois'][i]
            box = (x1,y1,x2-x1,y2-y1)
            objetos['tracker'][i].init(frame, box)
    
    return objetos
                   
def actualizarTrackers(objetos,frame):
    for i in range(len(objetos['ids'])):
        #hay que empezar a trackearlo porque lleva un frame sin aparecer
        if objetos['age'][i]>1:
            (success, box) = objetos['tracker'][i].update(frame)
            if success:
                objetos['rois'][i]=[box[1],box[0],box[1]+box[3],box[0]+box[2]]
                
    return objetos
                   












