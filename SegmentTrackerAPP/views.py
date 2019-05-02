from django.shortcuts import render
from django.http import HttpResponse
import os
import cv2
import mimetypes
from urllib.parse import urlparse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from SegmentTrackerAPP.models import *
from django.contrib.auth import authenticate, login
import json
import zipfile
import datetime
from SegmentTracker import settings
### IMPORTS PARA MOSTRAR IMÁGENES EN LA PÁGINA ####
from os import listdir
from os.path import isfile, join
#### IMPORTS PARA PREPROCESAMIENTO DE IMÁGENES ####
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import ntpath      
import time       
from . import Segmentador
from . import unet_CellSegmentation
###################################################

#imagen_gris = np.zeros((256,256), 'uint8')
#imagen_nueva = np.zeros((256,256,3), 'uint8')
@csrf_exempt
def load_images(dir):
    #mail = request.session['username']
    #idDataSet = request.session['idDataSet']
    #var query = "SELECT img.dir, img.name FROM segmenttrackerapp_image as img, segmenttrackerapp_usuario as us segmenttrackerapp_dataset as ds WHERE img.idDataSet = ds.id and ds.id_user = us.id and correo = " + mail + " and ds.id =" + idDataSet + ";" 

    #var x = Usuario.objects.raw(query) 
    
    #print("DATA IMAGES: ", x)
    time.sleep(2)

    onlyfiles = []
    for f in listdir(dir):
        if isfile:
            onlyfiles += [f]
    context = {
        'loaded_images' : onlyfiles,
    }

    return context
    
    
@csrf_exempt
def load_images2(id_ds):
    time.sleep(2)

    resultantes=ImageResult.objects.filter(idDataSet = id_ds).order_by('name')
    onlyfiles=[]
    for i in resultantes:
        camino = i.get_name()
        onlyfiles.append(camino)
    context = {
        'loaded_images' : onlyfiles,
        'id_ds': id_ds
    }

    return context
    
@csrf_exempt
def load_images3(id_ds):
    time.sleep(2)

    resultantes=ColoresResult.objects.filter(idDataSet = id_ds).order_by('name')
    onlyfiles=[]
    for i in resultantes:
        camino = i.get_name()
        onlyfiles.append(camino)
    context = {
        'loaded_images' : onlyfiles,
        'id_ds': id_ds
    }

    return context
    

def index(request):
    return render(request, 'SegmentTracker_WEB/login.html', {})

def process_image(request):
    dir = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(request.session["idDataSet"])
    context = load_images(dir) 
    #print("[CONTEXT]: ", context)
    return render(request, 'SegmentTracker_WEB/procesar_Imagenes.html', context)

def loadImage(request):
    dir = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(request.session["idDataSet"])
    dir = dir + "/" + request.GET.get("img")
    #print("[LOAD_IMAGE]: ", dir)
    data = open(dir, "rb").read()
    contentType = mimetypes.guess_type(dir)
    #print("CT:", contentType[0])
    return HttpResponse(data, content_type=contentType[0])

# carga las imagenes, pero recibe el id del dataset en vez de sacarlo
def loadImage2(request):
    id_ds = request.GET.get("id_ds")
    dir = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(id_ds)
    dir = dir + "/" + request.GET.get("img")
    #print("[LOAD_IMAGE]: ", dir)
    data = open(dir, "rb").read()
    contentType = mimetypes.guess_type(dir)
    #print("CT:", contentType[0])
    return HttpResponse(data, content_type=contentType[0])


def loadImageSeg(request):
    dir = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(request.session["idDataSet"]) + "_resultado"
    dir = dir + "/" + request.GET.get("img")
    #print("[LOAD_IMAGE]: ", dir)
    data = open(dir, "rb").read()
    contentType = mimetypes.guess_type(dir)
    #print("CT:", contentType[0])
    return HttpResponse(data, content_type=contentType[0])
    
def loadImageSeg2(request):
    id_ds = request.GET.get("id_ds")
    print("----------*--------------")
    print(id_ds)
    dir = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(id_ds) + "_resultado"
    dir = dir + "/" + request.GET.get("img")
    #print("[LOAD_IMAGE]: ", dir)
    data = open(dir, "rb").read()
    contentType = mimetypes.guess_type(dir)
    #print("CT:", contentType[0])
    return HttpResponse(data, content_type=contentType[0])


def load(request):
    
    if not request.session.has_key('username'):
        request.session['username'] = "guestUser"
    
    #print("[USER_NAME] ", request.session['username'])

    ##print("PATH >>>>>>>: ", request.)
    #print("[PATH_LOCAL]", os.getcwd())
    path = request.path
    #print("[PATH_DESTINO]: ", path)
    #print("[QUERY]:", request.GET) # Esto deja un diccionario con los parametros enviados
    #print("META:", request.POST) # Esto no lo entiendo muy bien
    #print("Path Local", os.getcwd())
    
    #print("DATOS:", datos)
    path = os.getcwd() + path.replace('SegmentTrackerAPP', "SegmentTrackerAPP/templates/SegmentTracker_WEB/")
    #print("[PATH_FIN]: ", path)
    #print("CONTENT_type:" + request.content_type)
    data = open(path, "rb").read()
    
    contentType = mimetypes.guess_type(path)
    #print("CT:", contentType[0])
    
    return HttpResponse(data, content_type=contentType[0])

@csrf_exempt
def upload_driver(request):
    base_folder = settings.PULL_DRIVER_UPLOAD_PATH 
    #print("META: ", request.META, type(request.META))
    #print("POST: ", request.POST.keys(), type(request.POST))
    #print("[REQUEST_FILES]: ", request.FILES, type(request.FILES))
    ret = {}
    keys = request.POST.keys()
    target_folder = base_folder + "/DataSet_" + str(request.session["idDataSet"]) # Para separar los Data Sets
    #print("TF: ", target_folder)
    if not os.path.exists(target_folder): 
        os.mkdir(target_folder)
    for name in keys:
        filename = request.POST[name]
        upload_file = request.FILES[name]
        if upload_file:
            target = os.path.join(target_folder, filename)
            with open(target, 'wb+') as dest:
                for c in upload_file.chunks():
                    dest.write(c)
            ret['file_remote_path'] = target
            image = Imagen(idDataSet=request.session["idDataSet"], dir = target_folder, name = filename)
            image.save()
        else:
            return HttpResponse(status = 500)
        
    return render(request, 'SegmentTracker_WEB/Cargar_imagenes.html', {})
    #return HttpResponse(json.dumps(ret), content_type = "application/json")

###### -PREPROCESAMIENRO DE IMaGENES- #######

def remove_image_file(request):
    x = request.GET.get('image')
    dir = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(request.session["idDataSet"]) + "/" + x
   
    if os.path.exists(dir):
        os.remove(dir)

def denoise_image(path, dest):
    ##print('PATH IMAGEN PREPROCESADA: '+path)
    ##print('PATH DESTINO IMAGEN: '+dest)
    img = cv2.imread(path)

    gaussian_3 = cv2.GaussianBlur(img, (9,9), 3.0)
    unsharp_image = cv2.addWeighted(img, 4, gaussian_3, -0.5, 0, img)
    new_image_name = dest + "//" + ntpath.basename(path)

    #CREA IMAGEN PREPROCESADA
    cv2.imwrite(new_image_name, unsharp_image)

def preprocess_image(request):
    #print("KDSKJSDKJSKJDKSJDKSJD**-*-*-*-*-*-*-*-*-")
    dir = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(request.session["idDataSet"])
    img = dir + "/" + request.GET.get('image')
    denoise_image(img,dir)
    #print("[PROCCESS_IMAGE]: ", request.GET.get('image'))


############### -CRUD USUARIO- ###############

def usermail_present(mail):
    if Usuario.objects.filter(correo=mail).exists():
        return True
    return False

def addUserToDataBase(request):
    name = request.GET.get('name')
    lastname = request.GET.get('lastname')
    mail = request.GET.get('email')
    institute = request.GET.get('institute')
    thepassword = request.GET.get('HsyRdIC0coC4rJu4')
    present = "NO"
    #print("[NEW_USER]: ", mail, "&",institute)
    #print("[User_is_already_signed]: ", mail)
    if(usermail_present(mail)):
        present = "YES"
        return HttpResponse(present, content_type = "text/plain")
    else: 
        u = Usuario(nombre=name, apellido=lastname, correo=mail, institucion=institute, password=thepassword)
        u.save()
        return HttpResponse(present, content_type = "text/plain")

def deteleUserFromDataBase(request):
    mail = request.GET.get('email')
    Usuario.objects.filter(correo=mail).delete()

def getUser(request):
    mail = request.GET.get('user_email')
    return Usuario.objects.raw("SELECT * FROM SegmentTrackerAPP_usuario WHERE correo = '" + mail + "';")

def updateUser(name, mail, institute, thepassword):
    Usuario.objects.filter(correo=mail).update(nombre=name, correo=mail, institucion=institute, password=thepassword)
    
def verHistorial(request):
    if(not bool(request.session)):
        return render(request, 'SegmentTracker_WEB/nueva_segmentacion.html', {})
    mail = request.session['username']
    if(not usermail_present(mail)):
        return render(request, 'SegmentTracker_WEB/nueva_segmentacion.html', {})
    # Usuario.objects.filter(correo=mail).update(nombre=name, correo=mail, institucion=institute, password=thepassword)
    username = request.session['username']
    cliente = Usuario.objects.get(correo = username)
    pasados = DataSet.objects.filter(id_user=cliente.getId())
    return render(request, 'SegmentTracker_WEB/Historial.html', {'anteriores': pasados})

def getPass(request):
    mail = request.GET.get('email')
    #print("EMAIL: ",mail)
    x = Usuario.objects.raw("SELECT * FROM SegmentTrackerAPP_usuario WHERE correo = '" + mail + "';")
    #print("X: ",x[0])

    request.session['username'] = mail
    request.session["id"] = x[0].id
    request.session["idDataSet"] = 1
    #print("[LOGGED_USER]: ", "YES")
    return HttpResponse(x[0].password, content_type = "text/plain")

def dataSetSession(request):
    name = "nombreSession"
    detail = request.GET.get('detail')
    
    #print("[UserName_Session]: ", request.session['username'])
    #print("[Detail_Session]: ", detail)
    #print("[Id_session]: ", request.session['id'])
    #print("[Date_session]: ", datetime.date.today())

    ds = DataSet(id_user=int(request.session['id']), date=datetime.date.today(), name=name, desc=detail)
    ds.save()
    #:", ds.id)
    request.session["idDataSet"] = ds.id
    #context = {'saludo' : 'hello-world'}
    return render(request, 'SegmentTracker_WEB/Cargar_imagenes.html', {})

def logout(request):
    try:
        #print("[Entry_with_guestUser]: ", mail)
        request.session['username'] = "guestUser"
    except:
        pass
    return render(request, 'SegmentTracker_WEB/index.html', {})

def loggedUser(request):
    if(not bool(request.session)):
        return HttpResponse("bad", content_type = "text/plain")
    mail = request.session['username']
    if(usermail_present(mail)):
        #print("[User_is_already_signed]: ", mail)
        return HttpResponse("ok", content_type = "text/plain")
    else:
        #print("[User_is_not_signed]: ", mail)
        return HttpResponse("bad", content_type = "text/plain")

    
############### - Segmentation - ###############

def doSegmentation(request):
    test_id = request.session["idDataSet"]
    dir = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(request.session["idDataSet"])
    dirRes = dir + "_resultado/"
    if not os.path.exists(dirRes):
        os.mkdir(dirRes)
    
    dirRes2 = dir + "_resultado2/"
    if not os.path.exists(dirRes2):
        os.mkdir(dirRes2)
    unet_CellSegmentation.predict_web2(dir + "/", dirRes, test_id)
    """
    Segmentador.predict(load=settings.PULL_DRIVER_UPLOAD_PATH + "/weights_CET.pth",
        n_channels = 1,
        n_classes = 3,
        dir_pred = dir + "/",
        dir_pred_2 = None,
        dir_gt = dir + "/",
        evaluation="MSE", 
        out=None,
        dest=dirRes)
    """
    return HttpResponse("fine", content_type = "text/plain")

def segmented_image(request):
    #id_ds = id_ds = request.GET.get("idds")
    id_ds = request.session["idDataSet"]
    dir = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(request.session["idDataSet"]) +"_resultado"
    context = load_images2(request.session["idDataSet"])
    #print("[CONTEXT]: ", context)
    return render(request, 'SegmentTracker_WEB/resultado_procesar_Imagenes.html', context)
    
def ver_resultados(request):
    id_ds = id_ds = request.GET.get("id_ds")
    #id_ds = request.session["idDataSet"]
    dir = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(request.session["idDataSet"]) +"_resultado"
    context = load_images2(id_ds)
    #print("[CONTEXT]: ", context)
    return render(request, 'SegmentTracker_WEB/resultado_procesar_Imagenes2.html', context)

def estadisticas(request):
    #loggedUser2(request)
    if(not bool(request.session)):
        return render(request, 'SegmentTracker_WEB/nueva_segmentacion.html', {})
    mail = request.session['username']
    if(not usermail_present(mail)):
        return render(request, 'SegmentTracker_WEB/nueva_segmentacion.html', {})
    nombre = request.GET.get("img")
    id_ds = request.GET.get("id_ds")
    #id_ds = request.GET.get("idds")
    #id_ds = request.session["idDataSet"]
    imagen = ImageResult.objects.get(idDataSet = id_ds, name=nombre)
    imagen = imagen.get_imagen_base()
    colores = ColoresResult.objects.get(id_imagen_base = imagen)
    celula = Celula.objects.filter(id_imagen = imagen)
    
    ocup = 0
    area_min = 1e9
    area_max = -1
    promedio = 0
    cant_celulas = 0
    for i in celula:
        cant_celulas+=1
        promedio+=i.get_area()
        ocup+=i.get_area()
        if(i.get_area()>area_max):
            area_max = i.get_area()
        if(i.get_area()<area_min):
            area_min = i.get_area()
    ocup=float(ocup)
    ocup = ocup*100.0/(256.0*256.0)
    area_promedio = float(promedio)/float(cant_celulas)
    context = {'id_ds':id_ds, 'celula': celula, 'colores': colores, 'ocupado':round(ocup, 2), 'minima': area_min, 'maxima': area_max, 'cantidad': cant_celulas, 'promedio': round(area_promedio, 2)}
    return render(request, 'SegmentTracker_WEB/estadisticas.html', context)
    
def rastreo(request):
    id_ds = request.GET.get("id_ds")
    imagenes_col = ColoresResult.objects.filter(idDataSet = id_ds).order_by('name')
    imagen = imagenes_col[0]
    context = {'imagen': imagen, 'id_ds': id_ds}
    return render(request, 'SegmentTracker_WEB/seleccionar_celula.html', context)

def fill_alg(imagen_gris, imagen_nueva, i, j):
    
    print(str(i)+" "+str(j))
    imagen_nueva[i, j, 0] = 255
    imagen_nueva[i, j, 1] = 0
    imagen_nueva[i, j, 2] = 255
    imagen_gris[i,j]=0
    print(imagen_nueva[i, j, 1])
    if i>0:
        #print(str(i-1)+" "+str(j)+" "+"i-1: "+str(imagen_gris[i-1, j]))
        if(imagen_gris[i-1, j]>60):
            
            imagen_gris[i-1,j]=0
            fill_alg(imagen_gris, imagen_nueva, i-1, j)
    if i<imagen_gris.shape[0]-1:
        # print(str(i+1)+" "+str(j)+" "+"i+1: "+str(imagen_gris[i+1, j]))
        if(imagen_gris[i+1, j]>60):
            
            imagen_gris[i+1,j]=0
            fill_alg(imagen_gris, imagen_nueva, i+1, j)#imagen_gris, imagen_nueva, 
    if j > 0:
        # print(str(i)+" "+str(j-1)+" "+"j-1: "+str(imagen_gris[i, j-1]))
        if(imagen_gris[i, j-1]>60):
            
            imagen_gris[i,j-1]=0
            fill_alg(imagen_gris, imagen_nueva, i, j-1)
    if j < imagen_gris.shape[1]-1:
        # print(str(i)+" "+str(j+1)+" "+"j+1: "+str(imagen_gris[i, j+1]))
        if(imagen_gris[i, j+1]>60):
            
            imagen_gris[i,j+1]=0
            fill_alg(imagen_gris, imagen_nueva, i, j+1)
            

def rastreo_iniciar(request):
    print("*********************")
    print("*********************")
    print("*********************")
    print("*********************")
    print("*********************")
    print("*********************")
    print("*********************")
    print("*********************")
    global imagen_nueva
    global imagen_gris
    
    id_ds = request.GET.get("id_ds")
    #id_celula = request.GET.get("id_celula")
    imagenes_col = ColoresResult.objects.filter(idDataSet = id_ds).order_by('name')
    imagen = imagenes_col[0]
    imagen_fuente = imagen.get_imagen_base()
    celula_inicio = Celula.objects.get(id_imagen = imagen_fuente, num_celula = id_celula)
    x_inicio = celula_inicio.get_x()
    y_inicio = celula_inicio.get_y()
    dataset = DataSet.objects.get(id = id_ds)
    todas_imagenes = Imagen.objects.filter(idDataSet = id_ds).order_by('name')
    imagenes_segmentadas = ImageResult.objects.filter(idDataSet = id_ds).order_by('name')
    coordenadas_pintar = []
    
    y_temp=1
    x_temp=1
    #coordenadas_pintar.append((x, y))
    for im_temp in todas_imagenes:
        celulas_imagen = Celula.objects.filter(id_imagen = im_temp)
        dist_max = 10000000000.00
        for cel_temp in celulas_imagen:
            if(cel_temp.distancia(x_inicio, y_inicio)<dist_max):
                dist_max=cel_temp.distancia(x_inicio, y_inicio)
                x_temp=cel_temp.get_x()
                y_temp=cel_temp.get_y()
        coordenadas_pintar.append((x_temp, y_temp))
        x_inicio = x_temp
        y_inicio = y_temp
    print("----------*-********---------")
    print(coordenadas_pintar)
    print("----------*-********&*&**&*&*&*&*&*+-+-+-+-+-")
    
    directorio_salida = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(id_ds)+ "_resultado/"
    cont=0
    numeros=[]
    nombres=[]
    primero="1"
    for coords in coordenadas_pintar:
        
        x = coords[0]
        y = coords[1]
        imagen_nueva = np.zeros((256,256,3), 'uint8')
        imagen_fuente = imagenes_segmentadas[cont]
        
        filename = imagen_fuente.get_full_path()
        #cargar la imagen
        im = Image.open(filename)
        im = im.convert('L')
        #im.show()
        im = im.resize((256, 256))
        im = np.array(im)
        imagen_gris = im
        fill_alg(imagen_gris, imagen_nueva, x, y)
        im_m = Image.fromarray(imagen_gris, 'L')
        #im_m.show()
        im_rgb = Image.fromarray(imagen_nueva, 'RGB')
        #im_rgb.show()
        
        nombre = str(imagen_fuente.get_imagen_base().get_id()).zfill(6)+"_"+str(id_celula).zfill(6)+ '_rastreo.png'
        im_rgb.save(os.path.join(directorio_salida, nombre))
        ImagenRastreo.objects.create(idDataSet = dataset, dir = directorio_salida, name = nombre, num_secuencia = cont, num_celula = id_celula)
        if(cont!=0):
            numeros.append(cont)
            nombres.append(nombre)
        else:
            primero=nombre
        cont+=1
        
    
    imagenes_col = ColoresResult.objects.filter(idDataSet = id_ds).order_by('name')
    imagen = imagenes_col[0]
    context = {'primero':primero, 'rastreos': nombres, 'id_ds': id_ds, 'copias': numeros}
    return render(request, 'SegmentTracker_WEB/resultado_rastreo.html', context)

def descargarimagenes(request):
    id_ds = request.GET.get("id_ds")
    dir_zip = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(id_ds) + "_resultado/"
    dir_zip = dir_zip+"imagenes_"+str(id_ds)+".zip"
    try:
        os.remove(dir_zip)
    except:
        print("file does not exists")
    nombre = "imagenes_"+str(id_ds)+".zip"
    zip = zipfile.ZipFile(dir_zip, 'a')
    imagenes = ImageResult.objects.filter(idDataSet = id_ds)
    for imagen in imagenes:
        zip.write(imagen.get_full_path())
    zip.close()
    data = open(dir_zip, "rb").read()
    response = HttpResponse(data, content_type='application/force-download')
    response['Content-Disposition'] = 'attachment; filename="%s"' % nombre
    return response
    
def descargarcsv(request):
    print("++++++++++++++++++++++++")
    id_ds = request.GET.get("id_ds")
    dir_zip = settings.PULL_DRIVER_UPLOAD_PATH + "/DataSet_" + str(id_ds) + "_resultado/"
    dir_zip = dir_zip+"csv_"+str(id_ds)+".zip"
    try:
        os.remove(dir_zip)
    except:
        print("file does not exists")

    nombre = "csv_"+str(id_ds)+".zip"
    zip = zipfile.ZipFile(dir_zip, 'a')
    imagenes = CSVResult.objects.filter(idDataSet = id_ds)
    for imagen in imagenes:
        zip.write(imagen.get_full_path())
    zip.close()
    data = open(dir_zip, "rb").read()
    #print("CT:", contentType[0])
    response = HttpResponse(data, content_type='application/force-download')
    response['Content-Disposition'] = 'attachment; filename="%s"' % nombre
    return response
    
def todorastreo(request):
    id_ds = id_ds = request.GET.get("id_ds")
    #id_ds = request.session["idDataSet"]
    resultantes=ColoresResult.objects.filter(idDataSet = id_ds).order_by('name')
    numeros=[]
    j=1
    onlyfiles=[]
    nombres=[]
    if(len(resultantes)>=1):
        primero=resultantes[0].get_name()
    for i in range(1, len(resultantes)):
        nombres.append(resultantes[i].get_name())
        numeros.append(j)
        j+=1        

    context = {'primero':primero, 'rastreos': nombres, 'id_ds': id_ds, 'copias': numeros}
    #print("[CONTEXT]: ", context)
    return render(request, 'SegmentTracker_WEB/resultado_rastreo2.html', context)




