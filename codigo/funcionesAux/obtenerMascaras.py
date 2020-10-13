import numpy as np
from itertools import groupby
import json
import numpy as np
from pycocotools import mask
from skimage import measure


#Decodificar la mascara del objeto del groundtruth
def mascaraObjeto(gtObjetoi):
    mascara={'size': gtObjetoi['size'], 'counts':gtObjetoi['counts'] }
    cositas=mask.decode(mascara)
    return cositas

#Devuelve la lista de mascaras del frame actual con el mismo formato que objetos['masks']
def obtenerMatrizMascaras(groundt,ignorarZonaGris):
    
    N=len(groundt)
    primeraVez=True
    
    #Si se hace pop, no queremos mirar el i+1, queremos mirar el i
    i=0
    #En todas las iteraciones hay que avanzar para recorrer los N elementos
    j=0
    
    #Hay que generar la matriz de mascaras de 3 dimensiones del frame actual
    while j < N:
         #Para solo detectar coches y puntos ciegos
         if groundt[i].class_id==1 or (not ignorarZonaGris and groundt[i].class_id==10):
             #decodificar mascara de RLE a matriz 
             maskObjetoi=mascaraObjeto(groundt[i].mask)
             #añadir la tercera dimension para crear una lista de matrices
             maskObjetoi=np.expand_dims(maskObjetoi,axis=2)
             
             #primera iteracion en la que masks hay que definirla con el primer objeto
             if primeraVez:
                masks=maskObjetoi
                primeraVez=False
             else:
                 #añadir matriz a la lista de matrices
                 masks=np.concatenate((masks,maskObjetoi),axis=2)
              
             i+=1;
                
         #si el objeto i se ignora hay que eliminarlo de la lista de groundtruth para acciones posteriores
         else:
             groundt.pop(i)
            
         j+=1
          
    if not primeraVez:
        return masks,groundt
    #Por si no entra en el if del while ninguna vez que no pete porque no se han declarado las variables masks,groundt
    else:
        return [],[]
        
