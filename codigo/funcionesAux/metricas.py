import numpy as np
from funcionesAux import visualize
from funcionesAux import obtenerMascaras
from munkres import Munkres, print_matrix

# paquetes para el tracker
import imutils
import time
import cv2

#Convertir en cuadrada la matriz de pesos para poder aplicar algoritmo hungaro
def cuadrificador(matrizPesos,valor):
    (a,b)=matrizPesos.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(matrizPesos,padding,mode='constant',constant_values=valor)


#Devolver la mejor correspondencia entre objetos
def algoritmoHungaro(matrizPesos):
    #hacer la matriz cuadrada
    matrizCuadrada=cuadrificador(matrizPesos,0)
    #convertirla en lista para poder aplicar el algoritmo hungaro
    matriz=matrizCuadrada.tolist()
    #para convertir el problema de max en min
    for i in range(len(matriz)): 
        for j in range(len(matriz)): 
            matriz[i][j] = 1-matriz[i][j]
  
    m = Munkres()
    indexes = m.compute(matriz)
    #print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    indices=[]
    for fila, col in indexes:
        indices+=[[fila,col]]
        
    return matrizCuadrada,indices


#Calculo de IoU entre mascaras 
def IoU_Mask(maskO,maskR):
    return np.sum(np.logical_and(maskO, maskR))/np.sum(np.logical_or(maskO, maskR))
    
#Calculo IoU entre bounding boxes   
def IoU_BB(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
  
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    
    return iou

#Asocia los objetos del frame actual con los vistos en frames anteriores
def objetosMostrar(modelRAM,objetos,r,ultimoID, colores, pesoMask, pesoRAM, verbose):
    lenObjetos=len(objetos['ids'])
    lenR=len(r['ids'])
    #matrizPesos[i][j]  i->objetos(objetos distintos detectados entre todos los frames) j->r(objetos del frame actual)
    pesosSolapamientoMask=np.empty((lenObjetos, lenR))
    pesosSolapamientoBB=np.empty((lenObjetos, lenR))
    pesosRam=np.empty((lenObjetos, lenR))
    objetosFrame=[]
    
    #--------------------------------------------------------------------------------------
    #Crear la matriz de pesos entre los objetos del frame actual y de los frames anteriores
    #--------------------------------------------------------------------------------------
    
    #FILA:objetos ya vistos
    for i in range(0,lenObjetos):

        y1, x1, y2, x2 = objetos['rois'][i]
        bb1={
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2,
        }
        
        #COLUMNA:objetos vistos en el frame actual
        for j in range(0,lenR):
            
            y1, x1, y2, x2 = r['rois'][j]
            bb2={
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
            }
            
            
            #Matriz con solapamiento entre máscaras
            pesosSolapamientoMask[i][j]=IoU_Mask(objetos['masks'][:,:,i],r['masks'][:,:,j])
            #Matriz con solapamiento entre bounding boxes
            pesosSolapamientoBB[i][j]=IoU_BB(bb1,bb2)
            
            #Matriz con lo que se parece cada par de objetos segun la CNN con descriptores aprendidos
            input1=np.expand_dims(objetos['ram'][i,:,:,:],axis=0)
            input2=np.expand_dims(r['ram'][j,:,:,:],axis=0)
            salida=modelRAM.predict([input1,input2], batch_size=10, verbose=0)
            #Probabilidad de que sea el mismo objeto segun la salida de la red
            pesosRam[i][j]=salida[0][0]
    
   
    #--------------------------------------------------------------------
    #Fusionar la información de solapamiento con la información de la CNN
    #--------------------------------------------------------------------
    
    matrizPesosFinal = np.zeros((lenObjetos,lenR))
    for i in range(lenObjetos):
        for j in range(lenR):
            
            #Si el objeto no aparece en el frame anterior se utiliza el solapamiento de bounding boxes
            if objetos['age'][i]>1:
                matrizPesosFinal[i][j]=pesosSolapamientoBB[i][j]*pesoMask+pesosRam[i][j]*pesoRAM
            #Si el objeto aparece en el frame anterior se utiliza el solapamiento de mascaras
            else:
                matrizPesosFinal[i][j]=pesosSolapamientoMask[i][j]*pesoMask+pesosRam[i][j]*pesoRAM
    
    #Algoritmo hungaro
    matrizCuadradaFinal, indicesFinal=algoritmoHungaro(matrizPesosFinal)
    
    #--------------------------------------------------------------------------------------------------
    #Actualizar información de los objetos según las asociaciones establecidas por el algoritmo hungaro
    #--------------------------------------------------------------------------------------------------
    
    #si se añade una fila con 0s para que sea cuadrada no hay problema, si se añade columna segunda condicion elif
    for fila,col in indicesFinal:
        
        #si es un objeto que ya existia en un frame anterior, se usa el id y color de dicho objeto
        if fila<len(objetos['ids']) and col<len(r['ids']) and matrizCuadradaFinal[fila][col] > 0:
            
            #asignar parametros del objeto ya visto
            r['ids'][col]=objetos['ids'][fila]
            r['colors'][col]=objetos['colors'][fila]
            
            #actualizar la posicion del objeto ya visto
            objetos['rois'][fila]=r['rois'][col]
            objetos['scores'][fila]=r['scores'][col]
            objetos['masks'][:,:,fila]=r['masks'][:,:,col]
            objetos['ram'][fila,:,:,:]=r['ram'][col,:,:,:]
            objetos['age'][fila]=0
            objetos['tracker'][fila]=cv2.TrackerCSRT_create()
            
            
        #es un objeto nuevo - segunda condicion para evitar columnas(r) que se han añadido para hacer la matriz cuadrada 
        elif col<len(r['rois']):
            #añadir nuevo color e id al nuevo objeto
            r['colors'][col]=colores[ultimoID]
            r['ids'][col]=ultimoID
            
            #añadir nuevo objeto a la lista de objetos encontrados
            objetos['rois'] = np.vstack((objetos['rois'], r['rois'][col])) 
            objetos['class_ids']=np.append(objetos['class_ids'],r['class_ids'][col])
            objetos['scores']=np.append(objetos['scores'],r['scores'][col])
            objetos['ids']=np.append(objetos['ids'],r['ids'][col])
            objetos['colors'].insert(len(objetos['colors']), r['colors'][col])
            nuevaMascara=r['masks'][:,:,col]
            nuevaMascara=np.expand_dims(nuevaMascara,axis=2)
            objetos['masks']=np.concatenate((objetos['masks'],nuevaMascara),axis=2)
            nuevaRam=r['ram'][col,:,:,:]  
            nuevaRam=np.expand_dims(nuevaRam,axis=0)
            objetos['ram']=np.concatenate((objetos['ram'],nuevaRam),axis=0)
            objetos['age'] = np.append(objetos['age'],0)
            objetos['tracker']=np.append(objetos['tracker'],cv2.TrackerCSRT_create())
               
            ultimoID+=1
           
    
 
    #lo que se debe mostrar en el frame actual 
    objetosFrame={
                "rois": r['rois'],
                "class_ids": r['class_ids'],
                "scores": r['scores'],
                "masks": r['masks'],
                "ids":r['ids'],
                "colors":r['colors'],   
                "ram": r['ram'],
                "age":r['age']
                }
    
    return objetos,objetosFrame,ultimoID

   
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
def actualizarMetricas(objetosFrame,groundt,datosMetricas,umbral,verbose):
    
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
            solape=IoU_Mask(objetosFrame['masks'][:,:,i],masksGt[:,:,j])
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
                matrizPesos[i][j]=IoU_Mask(objetosFrame['masks'][:,:,i],masksGt[:,:,j])

        #Aplicar alg hungaro dejando la matriz cuadrada
        matrizCuadrada,indices=algoritmoHungaro(matrizPesos)
        
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

#Borrar los objetos que no se han visto en numFrame frames
def borrarObjetosDesfasados(objetos,numFrames):
    lenObjetos=len(objetos['ids'])
    j=0 #solo pasar de indice si se ha anyadido un objeto
    y=0 #hay que recorrer el numero de objetos total, pero puede que no haya que cambiar de indice    
    while y<lenObjetos:
        #NO BORRO NI LAS A7 NI ROIS PORQUE REALMENTE NO LAS USO
        if objetos['age'][j]>=numFrames:
            objetos['class_ids']=np.delete(objetos['class_ids'],j)
            objetos['scores']=np.delete(objetos['scores'],j)
            objetos['ids']=np.delete(objetos['ids'],j)
            objetos['masks']=np.delete(objetos['masks'],j,axis=2)
            objetos['rois']=np.delete(objetos['rois'],j,axis=0)
            objetos['ram']=np.delete(objetos['ram'],j,axis=0)
            objetos['colors'].pop(j)
            objetos['age']=np.delete(objetos['age'],j)
        else:
            j+=1

        y+=1
        
    return objetos

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












