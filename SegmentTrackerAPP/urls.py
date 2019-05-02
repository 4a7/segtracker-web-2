from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^$', views.index, name = "index"),
    
    re_path(r'^files', views.upload_driver, name = "upload_driver"),
    re_path(r'^registrate', views.addUserToDataBase, name = "addUserToDataBase"),
    re_path(r'^signing', views.getPass, name = "getPass"),
    re_path(r'^DataSetSession', views.dataSetSession, name = "dataSetSession"),
    re_path(r'^logoutUser', views.logout, name = "logout"),
    re_path(r'^loggedUser', views.loggedUser, name = "loggedUser"),
    re_path(r'^resultado_procesar_Imagenes', views.segmented_image, name = "segmented_image"),
    re_path(r'^procesar_Imagenes', views.process_image, name = "process_image"),
    re_path(r'^loadImg', views.loadImage, name = "loadImage"),
    re_path(r'^loadImg2', views.loadImage2, name = "loadImage2"),
    re_path(r'^loadImageSeg', views.loadImageSeg, name = "loadImageSeg"),
    re_path(r'^loadIaSeg2', views.loadImageSeg2, name = "loadIaSeg2"),
    re_path(r'^remove_image', views.remove_image_file, name = "remove_image_file"),
    re_path(r'^do_denoising', views.preprocess_image, name = "preprocess_image"),
    re_path(r'^doSegmentation', views.doSegmentation, name = "doSegmentation"),
    re_path(r'^estadisticas', views.estadisticas, name = "estadisticas"),
    re_path(r'^rastreo', views.rastreo, name = "rastreo"),
    re_path(r'^historial', views.verHistorial, name = "historial"),
    re_path(r'^riniciar', views.rastreo_iniciar, name = "rastreo_iniciar"),
    re_path(r'^descargarimagenes', views.descargarimagenes, name = "descargarimagenes"),
    re_path(r'^descargacsv', views.descargarcsv, name = "descargacsv"),
    re_path(r'^ver_resultados', views.ver_resultados, name = "ver_resultados"),
    re_path(r'^todorastreo', views.todorastreo, name = "todorastreo"),
    re_path(r'^', views.load, name = "load"),
    
]
