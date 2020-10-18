# TFG
Trabajo Fin de Grado, Universidad de Zaragoza - Seguimiento y Segmentación de Múltiples Objetos con Descriptores Aprendidos (Multi-Object Tracking &amp; Segmentation with Learnt Descriptors)

![](prueba.gif)

Paquetes de Python necesarios
-----------------------------
Ver el archivo requirements.txt

Archivo adicional necesario
---------------------------
1) Descargar de https://github.com/matterport/Mask_RCNN/releases -> mask_rcnn_coco.h5
2) Colocarlo en TFG/codigo

¿Cómo utilizar el Sistema de Seguimiento y Segmentación de Múltiples Objetos con Descriptores Aprendidos?
-----------------------------------------------------------------------------------------------------------
1) Elije un vídeo sobre el que quieras realizar el Seguimiento y la Segmentación de coches ("prueba.mp4")
2) Coloca el vídeo en TFG/videos
2) Abre con jupyter notebook el fichero TFG/demo.ipynb y baja hasta el final hasta llegar a la función "procesarFrame"
4) Ejecuta la celda de dicha función pasándole como parámetro el nombre de tu video ("procesarFrame(prueba.mp4)")
5) En TFG/videosProcesados se generará "prueba.mp4" (video procesado) y "prueba" (carpeta que contiene cada frame procesado individualmente)
