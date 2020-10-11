from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn import preprocessing

def crearDatosTraining():
    
    #LEER .CSV CON LOS DATOS DE ENTRENAMIENTO
    rois=readRois()
    #print(rois.shape)
    ids=readIds()
    #print(ids.shape)

    idsSort=np.sort(ids)               #array ordenado de menor a mayor
    idsIndexSort=np.argsort(ids)       #indices que ocupan cada posicion para ordenar el array
    #print(ids)
    #print(idsSort)
    #print(idsIndexSort)

    rois=rois[idsIndexSort]             #ordenarlo usando los indices que indican el orden 
    #print(rois.shape)
    
    print('rois.shape->',rois.shape)


    #RANDOMIZAMOS EL ORDEN DE LO 10 FRAMES DE CADA OBJETO
    numObjetos=int(rois.shape[0]/10)
    for i in range(numObjetos):
        pos=i*10
        random.shuffle(rois[pos:pos+10])
        
    numObjetos1=100
    numObjetos2=105
       
    #Se han recopilado datos en dos tandas para que quede equilibrado el numero de label 0,1
    rois1=rois[:1000,:]
    rois2=rois[1000:,:]
    
    print('rois1.shape->',rois1.shape)
    print('rois2.shape->',rois2.shape)
    
    data1cero = np.array([])
    data2cero = np.array([])
    data1uno = np.array([])
    data2uno= np.array([])
    labelcero = np.array([])
    labeluno = np.array([])
    
    #-------------------
    #PARTE 1
    #-------------------
    
    #CREAR 3 PAREJAS DEL MISMO OBJETO (6/10 OBJETOS USADOS)
    data1uno=[rois1[0]]
    data2uno=[rois1[1]]
    data1uno=np.append(data1uno,[rois1[2]],axis=0)
    data2uno=np.append(data2uno,[rois1[3]],axis=0)
    data1uno=np.append(data1uno,[rois1[4]],axis=0)
    data2uno=np.append(data2uno,[rois1[5]],axis=0)
       
    
    labeluno=np.append(labeluno,[1])
    labeluno=np.append(labeluno,[1])
    labeluno=np.append(labeluno,[1])
    
   
    #ya hemos procesado el objeto 0
    for i in range(1,numObjetos1):
        pos=i*10
        
        data1uno=np.append(data1uno,[rois1[pos]],axis=0)
        data2uno=np.append(data2uno,[rois1[pos+1]],axis=0)
        data1uno=np.append(data1uno,[rois1[pos+2]],axis=0)
        data2uno=np.append(data2uno,[rois1[pos+3]],axis=0)
        data1uno=np.append(data1uno,[rois1[pos+4]],axis=0)
        data2uno=np.append(data2uno,[rois1[pos+5]],axis=0)

        labeluno=np.append(labeluno,[1])
        labeluno=np.append(labeluno,[1])
        labeluno=np.append(labeluno,[1])
    
    
    primeraVez=True
    #PAREJAS ENTRE DISTINTOS OBJETOS
    asociarCon=0
    numVeces=np.zeros(numObjetos1)
    for i in range(numObjetos1):
        if numVeces[i]<4:
            pos=i*10+6
            asociarCon=1
            while numVeces[i]<4:
                pos2=(i+asociarCon)*10+6
                #print("objeto,index -> " + str(i) + " , " + str(int(numVeces[i])))
                #print("objetoUnion,indexUnion -> " + str(i+asociarCon) + " , " + str(int(numVeces[i+asociarCon])))
                index1=int(pos+numVeces[i])
                index2=int(pos2+numVeces[i+asociarCon])
                if primeraVez:
                    data1cero=[rois1[index1]]
                    data2cero=[rois1[index2]]
                    primeraVez=False
                else:
                    data1cero=np.append(data1cero,[rois1[index1]],axis=0)
                    data2cero=np.append(data2cero,[rois1[index2]],axis=0)
                    
                numVeces[i]+=1                                         
                numVeces[i+asociarCon]+=1    
                asociarCon+=1   

                labelcero=np.append(labelcero,[0])
                
  
    #-------------------
    #PARTE 2
    #-------------------
    
    #CREAR 2 PAREJAS DEL MISMO OBJETO (4/10 OBJETOS USADOS)
    data1uno=np.append(data1uno,[rois2[0]],axis=0)
    data2uno=np.append(data2uno,[rois2[1]],axis=0)
    data1uno=np.append(data1uno,[rois2[2]],axis=0)
    data2uno=np.append(data2uno,[rois2[3]],axis=0)
        
    
    labeluno=np.append(labeluno,[1])
    labeluno=np.append(labeluno,[1])
   
    #ya hemos procesado el objeto 0
    for i in range(1,numObjetos2):
        pos=i*10
        
        data1uno=np.append(data1uno,[rois2[pos]],axis=0)
        data2uno=np.append(data2uno,[rois2[pos+1]],axis=0)
        data1uno=np.append(data1uno,[rois2[pos+2]],axis=0)
        data2uno=np.append(data2uno,[rois2[pos+3]],axis=0)
   
        labeluno=np.append(labeluno,[1])
        labeluno=np.append(labeluno,[1])
    
    #PAREJAS ENTRE DISTINTOS OBJETOS
    asociarCon=0
    numVeces=np.zeros(numObjetos2)
    for i in range(numObjetos2):
        if numVeces[i]<6:
            pos=i*10+4
            asociarCon=1
            while numVeces[i]<6:
                pos2=(i+asociarCon)*10+4
                #print("objeto,index -> " + str(i) + " , " + str(int(numVeces[i])))
                #print("objetoUnion,indexUnion -> " + str(i+asociarCon) + " , " + str(int(numVeces[i+asociarCon])))
                index1=int(pos+numVeces[i])
                index2=int(pos2+numVeces[i+asociarCon])
                data1cero=np.append(data1cero,[rois2[index1]],axis=0)
                data2cero=np.append(data2cero,[rois2[index2]],axis=0)
                numVeces[i]+=1                                         
                numVeces[i+asociarCon]+=1    
                asociarCon+=1   

                labelcero=np.append(labelcero,[0])
    
    #Desordenar los datos para que se mezclen los objetos entre training y test  
    data1cero,data2cero=shuffle(data1cero,data2cero)
    data1uno,data2uno=shuffle(data1uno,data2uno)

    #515 label 0
    #510 label 1
    
    #TRAIN
    #---------------------------
    #70% parejas con label 0
    train_data1=data1cero[:360]
    train_data2=data2cero[:360]
    train_label=labelcero[:360]
 
    #70% parejas con label 1
    train_data1=np.append(train_data1,data1uno[:357],axis=0)
    train_data2=np.append(train_data2,data2uno[:357],axis=0)
    train_label=np.append(train_label,labeluno[:357],axis=0)
    
    #TEST
    #---------------------------
    #30% parejas con label 0
    test_data1=data1cero[360:]
    test_data2=data2cero[360:]
    test_label=labelcero[360:]
    #30% parejas con label 1
    test_data1=np.append(test_data1,data1uno[357:],axis=0)
    test_data2=np.append(test_data2,data2uno[357:],axis=0)
    test_label=np.append(test_label,labeluno[357:],axis=0)
    
    #Desordenar los datos para que se mezclen los 1s y 0s
    train_data1,train_data2,train_label=shuffle(train_data1,train_data2,train_label)
    test_data1,test_data2,test_label=shuffle(test_data1,test_data2,test_label)
    
    return train_data1, train_data2, train_label, test_data1, test_data2, test_label


#Guardar en un fichero los datos de entrenamiento listos para entrenar la red
def etiquetarYguardarDatos(ruta):
    #Leer los datos de entrenamiento del fichero csv
    train_data1, train_data2, train_label, test_data1, test_data2, test_label=crearDatosTraining() 

    f= open(ruta+'/train_data1.csv',"a")
    data=asarray(train_data1)
    savetxt(f, data, delimiter=',')
    f.close()

    f= open(ruta+'/train_data2.csv',"a")
    data=asarray(train_data2)
    savetxt(f, data, delimiter=',')
    f.close()

    f= open(ruta+'/train_label.csv',"a")
    data=asarray(train_label)
    savetxt(f, data, delimiter=',')
    f.close()
    
    f= open(ruta+'/test_data1.csv',"a")
    data=asarray(test_data1)
    savetxt(f, data, delimiter=',')
    f.close()

    f= open(ruta+'/test_data2.csv',"a")
    data=asarray(test_data2)
    savetxt(f, data, delimiter=',')
    f.close()

    f= open(ruta+'/test_label.csv',"a")
    data=asarray(test_label)
    savetxt(f, data, delimiter=',')
    f.close()



#Cargar datos listos para entrenar la red
def cargarDatos(ruta):
    f= open(ruta+'/train_data1.csv',"r")
    train_data1 = loadtxt(f, delimiter=',') 
    f.close()

    f= open(ruta+'/train_data2.csv',"r")
    train_data2 = loadtxt(f, delimiter=',') 
    f.close()

    f= open(ruta+'/train_label.csv',"r")
    train_label = loadtxt(f, delimiter=',') 
    f.close()
    
    f= open(ruta+'/test_data1.csv',"r")
    test_data1 = loadtxt(f, delimiter=',') 
    f.close()

    f= open(ruta+'/test_data2.csv',"r")
    test_data2 = loadtxt(f, delimiter=',') 
    f.close()

    f= open(ruta+'/test_label.csv',"r")
    test_label = loadtxt(f, delimiter=',') 
    f.close()

    #training data de (500,50176) -> (500,2,14,14,256)
    train_data1=np.reshape(train_data1,(train_data1.shape[0],14,14,256))
    train_data2=np.reshape(train_data2,(train_data2.shape[0],14,14,256))
    print(train_data1.shape)
    print(train_data2.shape)
    print(train_label.shape)
    
    #test data de (500,50176) -> (500,2,14,14,256)
    test_data1=np.reshape(test_data1,(test_data1.shape[0],14,14,256))
    test_data2=np.reshape(test_data2,(test_data2.shape[0],14,14,256))
    print(test_data1.shape)
    print(test_data2.shape)
    print(test_label.shape)
    
    return train_data1, train_data2, train_label, test_data1, test_data2, test_label
    
    
    
def saveRoi(roi):
    train_data=roi.reshape([1,14*14*256])
    f= open('trainingValues.csv',"a")
    data=asarray(train_data)
    savetxt(f, data, delimiter=',')

    f.close()

def saveId(id):
    f= open('trainingIds.csv',"a")
    data=asarray([id])
    savetxt(f, data, delimiter=',')

    f.close()
    
def readRois():
    f= open('samplesTesting/trainingValues.csv',"r")
    rois = loadtxt(f, delimiter=',')
    
    return rois

    f.close()
    
def readIds():
    f= open('samplesTesting/trainingIds.csv',"r")
    ids = loadtxt(f, delimiter=',')
    
    return ids

    f.close()
    
    
    
    
    
#PRUEBA DE QUE SE CREAN BIEN LOS DATOS DE ENTRENAMIENTO
'''
def crearDatosTraining():
    
    #LEER .CSV CON LOS DATOS DE ENTRENAMIENTO
    rois=readRois()
    #print(rois.shape)
    ids=readIds()
    #print(ids.shape)

    idsSort=np.sort(ids)               #array ordenado de menor a mayor
    idsIndexSort=np.argsort(ids)       #indices que ocupan cada posicion para ordenar el array
    #print(ids)
    #print(idsSort)
    #print(idsIndexSort)

    rois=rois[idsIndexSort]             #ordenarlo usando los indices que indican el orden 
    #print(rois.shape)
    
    print('rois.shape->',rois.shape)

    
    #PREPROCESAMIENTO DE LOS DATOS (ESTANDARIZACIÓN DE LOS DATOS)
    print('Mean:', round(rois[:,0].mean()))
    print('Standard deviation:', rois[:,0].std())
    # Create scaler
    scaler = preprocessing.StandardScaler()
    # Transform the feature
    rois = scaler.fit_transform(rois)
    # Print mean and standard deviation
    print('Mean:', round(rois[:,0].mean()))
    print('Standard deviation:', rois[:,0].std())
    
    
    #RANDOMIZAMOS EL ORDEN DE LO 10 FRAMES DE CADA OBJETO
    numObjetos=int(rois.shape[0]/10)
    for i in range(numObjetos):
        pos=i*10
        random.shuffle(rois[pos:pos+10])
        
    numObjetos1=100
    numObjetos2=105
        
    rois1=rois[:1000,:]
    rois2=rois[1000:,:]
    
    print('rois1.shape->',rois1.shape)
    print('rois2.shape->',rois2.shape)
    
    train_data1 = []
    train_data2 = []
    train_label = []
    
    #-------------------
    #PARTE 1
    #-------------------
    
    #CREAR 3 PAREJAS DEL MISMO OBJETO (6/10 OBJETOS USADOS)
    train_data1=[rois1[0]]
    train_data2=[rois1[1]]
    train_data1=np.append(train_data1,[rois1[2]],axis=0)
    train_data2=np.append(train_data2,[rois1[3]],axis=0)
    train_data1=np.append(train_data1,[rois1[4]],axis=0)
    train_data2=np.append(train_data2,[rois1[5]],axis=0)
       
    
    train_label.append(1)
    train_label.append(1)
    train_label.append(1)
    
   
    #ya hemos procesado el objeto 0
    for i in range(1,numObjetos1):
        pos=i*10
        
        train_data1=np.append(train_data1,[rois1[pos]],axis=0)
        train_data2=np.append(train_data2,[rois1[pos+1]],axis=0)
        train_data1=np.append(train_data1,[rois1[pos+2]],axis=0)
        train_data2=np.append(train_data2,[rois1[pos+3]],axis=0)
        train_data1=np.append(train_data1,[rois1[pos+4]],axis=0)
        train_data2=np.append(train_data2,[rois1[pos+5]],axis=0)

        train_label.append(1)
        train_label.append(1)
        train_label.append(1)
        
    
    #PAREJAS ENTRE DISTINTOS OBJETOS
    asociarCon=0
    numVeces=np.zeros(numObjetos1)
    for i in range(numObjetos1):
        if numVeces[i]<4:
            pos=i*10+6
            asociarCon=1
            while numVeces[i]<4:
                pos2=(i+asociarCon)*10+6
                #print("objeto,index -> " + str(i) + " , " + str(int(numVeces[i])))
                #print("objetoUnion,indexUnion -> " + str(i+asociarCon) + " , " + str(int(numVeces[i+asociarCon])))
                index1=int(pos+numVeces[i])
                index2=int(pos2+numVeces[i+asociarCon])
                train_data1=np.append(train_data1,[rois1[index1]],axis=0)
                train_data2=np.append(train_data2,[rois1[index2]],axis=0)
                numVeces[i]+=1                                         
                numVeces[i+asociarCon]+=1    
                asociarCon+=1   

                train_label.append(0)
                
  
    #-------------------
    #PARTE 2
    #-------------------
    
    #CREAR 2 PAREJAS DEL MISMO OBJETO (4/10 OBJETOS USADOS)
    train_data1=np.append(train_data1,[rois2[0]],axis=0)
    train_data2=np.append(train_data2,[rois2[1]],axis=0)
    train_data1=np.append(train_data1,[rois2[2]],axis=0)
    train_data2=np.append(train_data2,[rois2[3]],axis=0)
        
    
    train_label.append(1)
    train_label.append(1)
   
    #ya hemos procesado el objeto 0
    for i in range(1,numObjetos2):
        pos=i*10
        
        train_data1=np.append(train_data1,[rois2[pos]],axis=0)
        train_data2=np.append(train_data2,[rois2[pos+1]],axis=0)
        train_data1=np.append(train_data1,[rois2[pos+2]],axis=0)
        train_data2=np.append(train_data2,[rois2[pos+3]],axis=0)
   
        train_label.append(1)
        train_label.append(1)
    
    #PAREJAS ENTRE DISTINTOS OBJETOS
    asociarCon=0
    numVeces=np.zeros(numObjetos2)
    for i in range(numObjetos2):
        if numVeces[i]<6:
            pos=i*10+4
            asociarCon=1
            while numVeces[i]<6:
                pos2=(i+asociarCon)*10+4
                #print("objeto,index -> " + str(i) + " , " + str(int(numVeces[i])))
                #print("objetoUnion,indexUnion -> " + str(i+asociarCon) + " , " + str(int(numVeces[i+asociarCon])))
                index1=int(pos+numVeces[i])
                index2=int(pos2+numVeces[i+asociarCon])
                train_data1=np.append(train_data1,[rois2[index1]],axis=0)
                train_data2=np.append(train_data2,[rois2[index2]],axis=0)
                numVeces[i]+=1                                         
                numVeces[i+asociarCon]+=1    
                asociarCon+=1   

                train_label.append(0)
                
    #Desordenar los datos de entrenamiento    
    train_data1,train_data2,train_label=shuffle(train_data1,train_data2,train_label)
       
    return np.array(train_data1), np.array(train_data2), np.array(train_label)
'''
