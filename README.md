# TFG
Trabajo Fin de Grado, Universidad de Zaragoza - Seguimiento y Segmentación de Múltiples Objetos con Descriptores Aprendidos (Multi-Object Tracking &amp; Segmentation with Learnt Descriptors)

![](prueba.gif)

Antes de utilizar el sistema:
----------------------------------------------
  
  1) Probado con la versión de python 3.6. 
  
     No utilizar versiones posteriores.

  2) Instalar paquetes de Python:
  
    pip install -r requirements.txt

  3) AÑadir archivo necesario para evitar tener que entrenar Mask R-CNN:
  
    1) Descargar https://github.com/matterport/Mask_RCNN/releases -> mask_rcnn_coco.h5
    2) Colocarlo en TFG/codigo

  4) Añadir Funciones de COCO necesarias
  
    Desde carpeta TFG:
      
      pip install opencv-contrib-python
      cd codigo/coco/cocoapi/PythonAPI
      python setup.py build_ext install
      rm -rf build

¿Cómo utilizar el Sistema de Seguimiento y Segmentación de Múltiples Objetos con Descriptores Aprendidos?
-----------------------------------------------------------------------------------------------------------
1) Elije un vídeo sobre el que quieras realizar el Seguimiento y la Segmentación de coches ("prueba.mp4")
2) Coloca el vídeo en TFG/videos
2) Abre con jupyter notebook el fichero TFG/demo.ipynb y baja hasta el final hasta llegar a la función "procesarFrame"
4) Ejecuta la celda de dicha función pasándole como parámetro el nombre de tu video ("procesarFrame(prueba.mp4)")
5) En TFG/videosProcesados se generará "prueba.mp4" (video procesado) y "prueba" (carpeta que contiene cada frame procesado individualmente)
