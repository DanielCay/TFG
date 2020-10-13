import numpy as np
from funcionesAux import visualize
from funcionesAux import obtenerMascaras
from funcionesAux import asociarObjetos
from munkres import Munkres, print_matrix

# paquetes para el tracker
import imutils
import time
import cv2
   
#Devuelve el indice del objeto dentro del vector objetosFrame
#En el primer frame encontrado siempre false ya que todos los objetos del groundtruth tendran None
def buscarObjetoPorId(id, objetosFrame):
    encontrado=False
    i=0
    indice=0

    while not encontrado and i<len(objetosFrame['ids']):
        if objetosFrame['ids'][i] == id:
            encontrado=True
            indice=i
        i+=1

    return indice,encontrado
                 
#Actualiza la informacion necesaria para calcular MOTSA, MOTSP y sMOTSA con el frame actual
#Compara mascaras del gt con las mascaras del tracking del frame actual
def actualizarMetricas(objetosFrame,groundt,datosMetricas,umbral):
    
    IDS=0
    TP=0
    TPS=0
    FP=0
    FN=0
    M=0
    obj=datosMetricas['objetos']  #obj[0]=objeto del gt con track_id==0
    
    #-------------------------------------------------------------------------------------------------------
    #1) SE ELIMINAN LOS OBJETOS QUE ESTEN EN UN PUTNO CIEGO DE LA LISTA DE OBJETOS DETECTADOS (OBJETOSFRAME)
    #-------------------------------------------------------------------------------------------------------
    
    #Calcular la matriz de mascaras con el mismo formato que objetosFrame obteniendo tambien puntos ciegos
    masksGt,groundt=obtenerMascaras.obtenerMatrizMascaras(groundt,False)

    #Puede contener objetos que se van a quitar porque estan donde el gt ve un punto ciego
    dim=len(objetosFrame['ids'])
    #Puede contener puntos ciegos
    dimGt=len(groundt)
    
  
    puntosCiegosObjetosFrame=[]
    #Para no meter mas veces en la lista el mismo elemento (si esta una vez ya se sabe que hay que borrarlo)
    primeraVez=True
    puntoCiego=True
    mejorPuntoCiego=0
    mejorAsociacion=0
    #Calculamos la matriz de pesos que contiene el solapamiento entre cada par de objetos(frame-groundtruth)
    #frame-Filas 
    for i in range(0,dim):
        #groundtruth-Columnas
        for j in range(0,dimGt):
            solape=asociarObjetos.IoU_Mask(objetosFrame['masks'][:,:,i],masksGt[:,:,j])
            if groundt[j].class_id==10:
                if solape>mejorPuntoCiego:
                    mejorPuntoCiego=solape
            else:
                if solape>mejorAsociacion:
                    mejorAsociacion=solape
                    
        #Solo se cuenta como objeto que está en punto ciego si se solapa mas con un punto ciego que con un objeto del gt
        if mejorPuntoCiego>mejorAsociacion:
            puntosCiegosObjetosFrame.append(objetosFrame['ids'][i])   
            
        mejorPuntoCiego=0
        mejorAsociacion=0   
  
    #Se borran los puntos ciegos de la lista de objetos detectados en el frame actual
    #Si no se han pasado copias de las listas a estas funcion desde demo, al hacer esto, metricas.objetosMostrar fallara
    for i in range(len(puntosCiegosObjetosFrame)):   
        pos,encontrado=buscarObjetoPorId(puntosCiegosObjetosFrame[i],objetosFrame)
        objetosFrame['ids']=np.delete(objetosFrame['ids'],pos)
        objetosFrame['masks']=np.delete(objetosFrame['masks'],pos,2)
        
    if verbose:
        print('objetosPuntoCiego->',len(puntosCiegosObjetosFrame))
   
       
    #Calcular la matriz de mascaras con el mismo formato que objetosFrame eliminando puntos ciegos
    masksGt,groundt=obtenerMascaras.obtenerMatrizMascaras(groundt,True)
    
    dim=len(objetosFrame['ids'])
    dimGt=len(groundt)
  
    #-------------------------------------------------------------------------------------------------------
    #2)EMPAREJAR OBJETOS CON GROUNDTRUTH
    #-------------------------------------------------------------------------------------------------------    
    
    #Solo hay libres de objetosFrame(tracking) asi que se suma todos los que esten libres
    if dim>0 and dimGt==0:
        FP+=dim
    #Solo hay libres de groundt(groundtruth) asi que se suma todos los que esten libres
    elif dim==0 and dimGt>0:
        FN+=dimGt
    #No tiene sentido calcular nuevos pares si sigue habiendo solapamiento entre todos del frame anterior
    elif dim>0 and dimGt>0:
        matrizPesos=np.empty((dim, dimGt))

        #Calculamos la matriz de pesos que contiene el solapamiento entre cada par de objetos(frame-groundtruth)
        #fila->maskRCNN
        for i in range(0,dim):
            #columna->groundtruth
            for j in range(0,dimGt):
                matrizPesos[i][j]=asociarObjetos.IoU_Mask(objetosFrame['masks'][:,:,i],masksGt[:,:,j])

        #Aplicar alg hungaro dejando la matriz cuadrada
        matrizCuadrada,indices=asociarObjetos.algoritmoHungaro(matrizPesos)
        
        for fila,col in indices:
            if fila<dim and col<dimGt and matrizCuadrada[fila][col]>umbral:
                TP+=1 
                TPS+=matrizCuadrada[fila][col]

                #Si es la primera vez se asigna directamente el mejor objeto del frame actual
                if obj[groundt[col].track_id%100] == None:
                    obj[groundt[col].track_id%100]=objetosFrame['ids'][fila]
                else:
                    #Si se llega aqui es que se esta asociando un objeto distinto al del frame anterior
                    if obj[groundt[col].track_id%100] != objetosFrame['ids'][fila]:
                        IDS+=1
                        obj[groundt[col].track_id%100]=objetosFrame['ids'][fila]

            #si la fila o columna de la matriz se sale del rango de los respectivos vectores, 
            #hay que ignorar dichas posiciones de la matriz, porque son posiciones puestas para que sea cuadrada
            else:
                #Si la fila de la matriz esta dentro del rango del vector de objetos del frame, es un falso positivo
                if fila<dim:
                    FP+=1
                #Si la col de la matriz esta dentro del rango del vector de objetos del groundtruth, es un miss
                if col<dimGt:
                    FN+=1
    
    if verbose:          
        print('IDS -> ' + str(IDS))
        print('TP -> ' + str(TP))
        print('TPS -> ' + str(TPS))
        print('FP -> ' + str(FP))
        print('FN -> ' + str(FN))
        print('M -> ' + str(dimGt))
    

    datosMetricas={
         "objetos": obj,      #objetos reales resultado del groundtruth
         "IDS":datosMetricas['IDS']+IDS,
         "TP":datosMetricas['TP']+TP,
         "TPS":datosMetricas['TPS']+TPS,
         "FP":datosMetricas['FP']+FP,
         "FN":datosMetricas['FN']+FN,
         "M":datosMetricas['M']+dimGt
    }

    
    return datosMetricas

                   
#METRICA MOTSP
def calcularMOTSP(datosMetricas):
    return datosMetricas['TPS']/datosMetricas['TP']

#METRICA MOTSA
def calcularMOTSA(datosMetricas):
    return (datosMetricas['TP']-datosMetricas['FP']-datosMetricas['IDS'])/datosMetricas['M']
    #return 1-(datosMetricas['FN']+datosMetricas['FP']+datosMetricas['IDS'])/datosMetricas['M']

#METRICA sMOTSA
def calcularSMOTSA(datosMetricas):
    return (datosMetricas['TPS']-datosMetricas['FP']-datosMetricas['IDS'])/datosMetricas['M']












