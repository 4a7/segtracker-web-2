#!/usr/bin/env python

# Pattern Recognition and Machine Learning (PARMA) Group
# School of Computing, Costa Rica Institute of Technology
#
# title           :unet_CellSegmentation.py
# description     :Cell segmentation using pretrained unet architecture.
# authors         :Willard Zamora wizaca23@gmail.com,
#                 Manuel Zumbado manzumbado@ic-itcr.ac.cr
# date            :20180823
# version         :0.1
# usage           :python unet_CellSegmentation.py
# python_version  :>3.5
# ==============================================================================
#
import os
import time
import numpy as np
import sys
import copy
import math

from PIL import Image, ImageDraw, ImageFont
import glob
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
import pandas as pd
from pandas import DataFrame
#from flask_sqlalchemy import SQLAlchemy
import numpy as np
#from flask import Flask, render_template, request, send_from_directory
#from flask import Response, session, url_for, redirect
#from flask_dropzone import Dropzone
#from flask_uploads import UploadSet, configure_uploads
#from flask_uploads import IMAGES, patch_request_class
from PIL import Image
from pathlib import Path
#import flask
from scipy.spatial import distance
# import server
#import modelos
from .models import *

DOWNLOAD_DIRECTORY = "files"
#app = Flask(__name__)
#dropzone = Dropzone(app)
#servidor = 'mysql+pymysql://calidad:ss@localhost:3306/calidad_v1'
#app.config['SQLALCHEMY_DATABASE_URI'] = servidor
#db = SQLAlchemy(app)

# Se configura dropzone
#app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
#app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
#app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
#app.config['DROPZONE_REDIRECT_VIEW'] = 'exito'

# se configura uploads
#app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'

# se configura la clave del api
#app.config['SECRET_KEY'] = 'aire'

# Set channel configuration for backend
K.set_image_data_format('channels_last')
sys.setrecursionlimit(66000)

# Image size
img_rows = 256
img_cols = 256
# se guardan los codigos de los colores 
colores_r = [255, 0, 0, 255, 128, 0, 0, 0, 255, 128, 0, 166, 131, 180, 218]
colores_g = [0, 255, 0, 0, 0, 128, 0, 255, 0, 0, 128, 208, 214, 71, 121]
colores_b = [0, 0, 255, 255, 0, 0, 128, 255, 255, 128, 0, 92, 167, 209, 144]
# Dice coeficient parameter
smooth = 1.
# Paths declaration
image_path = 'imgtests\\*.png'    # 'raw/hoechst/test/*.png'
weights_path = '/home/j/Desktop/segtracker/weights/pre_0_3_5.h5'
pred_dir = 'preds\\'


# Compute dice coeficient used in loss function
def dice_coef(y_true, y_pred):
    """
    Funcion que calcula el coeficiente de dice
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f)
                                           + smooth)

def dice_manual(archivo, id_gt):
    print(id_gt)
    file = db.session.query(
            modelos.Archivo).filter(modelos.Archivo.id_archivo == id_gt).first()
    print(file.nombre)
    archivo.save(archivo.filename)
    im = Image.open(archivo.filename)
    im2 = Image.open(file.nombre)
    im = im.convert('1')
    im2 = im2.convert('1')
    im = np.array(im)
    im2 = np.array(im2)
    im = im.flatten()
    im2 = im2.flatten()
    im = im.tolist()
    im2 = im2.tolist()
    val = distance.dice(im, im2)
    print (val)
    return str(val)

def dice_manual2(nom1, nom2):
    im = Image.open(nom1)
    im2 = Image.open(nom2)
    im = im.convert('1')
    im2 = im2.convert('1')
    im = np.array(im)
    im2 = np.array(im2)
    im = im.flatten()
    im2 = im2.flatten()
    im = im.tolist()
    im2 = im2.tolist()
    val = distance.dice(im, im2)
    return val

def fill_alg(imagen_gris, imagen_color, color, i, j, cont, cords_x, cords_y):
    imagen_color[i, j, 0] = colores_r[color]
    imagen_color[i, j, 1] = colores_g[color]
    imagen_color[i, j, 2] = colores_b[color]
    cont+=1
    cords_x.append(i)
    cords_y.append(j)
    #print(str(i)+" "+str(j)+" "+str(imagen_gris[i,j]))
    imagen_gris[i,j]=0
    if i>0:
        if(imagen_gris[i-1, j]>60):
            #print(str(i-1)+" "+str(j)+" "+"i-1: "+str(imagen_gris[i-1, j]))
            imagen_gris[i-1,j]=0
            cont = fill_alg(imagen_gris, imagen_color, color, i-1, j, cont, cords_x, cords_y)
    if i<imagen_gris.shape[0]-1:
        if(imagen_gris[i+1, j]>60):
            #print(str(i+1)+" "+str(j)+" "+"i+1: "+str(imagen_gris[i+1, j]))
            imagen_gris[i+1,j]=0
            cont = fill_alg(imagen_gris, imagen_color, color, i+1, j, cont, cords_x, cords_y)
    if j > 0:
        if(imagen_gris[i, j-1]>60):
            # print(str(i)+" "+str(j-1)+" "+"j-1: "+str(imagen_gris[i, j-1]))
            imagen_gris[i,j-1]=0
            cont = fill_alg(imagen_gris, imagen_color, color,  i, j-1, cont, cords_x, cords_y)
    if j < imagen_gris.shape[1]-1:
        if(imagen_gris[i, j+1]>60):
            # print(str(i)+" "+str(j+1)+" "+"j+1: "+str(imagen_gris[i, j+1]))
            imagen_gris[i,j+1]=0
            cont = fill_alg(imagen_gris, imagen_color, color, i, j+1, cont, cords_x, cords_y)
    #im_rgb = Image.fromarray(imagen_color, 'RGB')
    #im_rgb.save(os.path.join(pred_dir,  '1_pred.png'))
    return cont




# Loss function
def dice_coef_loss(y_true, y_pred):
    """
    Funcion de error
    """
    return -dice_coef(y_true, y_pred)


# Load test data from directory
def load_test_data(image_path):
    """
    Funcion que se encarga de cargar los datos de prueba
    """
    raw = []
    image_filename = dict()
    count = 0
    for filename in glob.glob(image_path):
        name = os.path.basename(filename)[:-4]
        try:
            im = Image.open(filename)
            im = im.convert('L')
            im = im.resize((img_rows, img_cols))
            raw.append(np.array(im))
            image_filename[count] = name
            count += 1
            im.close()
        except IOError:
            print('Error loading image ', filename)
    return [raw, image_filename]


def get_file_names(image_path):
    """
    Funcion que retorna una lista con los nombres de archivo
    """
    image_filename = []
    for filename in glob.glob(image_path):
        name = os.path.basename(filename)[:-4]
        image_filename.append(name)
    return image_filename


# Preprocess loaded images
def preprocess(imgs):
    """
    Funcion que se encarga de preprocesar las imagenes
    Recibe como entrada un arreglo con las imagenes
    """
    imgs_p = np.ndarray((len(imgs), img_rows, img_cols), dtype=np.float32)
    for i in range(len(imgs)):
        imgs_p[i] = imgs[i].reshape((img_rows, img_cols))/255.

    imgs_p = imgs_p[..., np.newaxis]

    # Perform data normalization
    mean = imgs_p.mean()
    std = imgs_p.std()
    imgs_p -= mean
    imgs_p /= std

    return imgs_p


# Define unet architecture
def get_unet():
    """
    Funcion que crea el modelo
    """
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2),
                                       padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2),
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2),
                                       padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2),
                                       padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss,
                  metrics=[dice_coef])

    return model


def predict():
    start_time = time.time()
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    # Load test data
    cell_segmentation_data = load_test_data(image_path)

    # Preprocess and reshape test data
    x_test = preprocess(cell_segmentation_data[0])
    test_id = cell_segmentation_data[1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # Get model
    model = get_unet()

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    # Load weights
    model.load_weights(weights_path)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    # Make predictions
    imgs_mask_predict = model.predict(x_test, verbose=1)

    print('-' * 30)
    print('Saving predicted masks to files...')
    np.save('imgs_mask_predict.npy', imgs_mask_predict)
    print(imgs_mask_predict.shape)
    print('-' * 30)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    # Save predictions as images
    for image_pred, index in zip(imgs_mask_predict, range(x_test.shape[0])):
        image_pred = image_pred[:, :, 0]
        li = image_pred.shape[0]
        lj = image_pred.shape[1]
        image_pred[image_pred > 0.5] *= 255.
        # para colorear la imagen
        color = 0
        cont = 0
        cords_x = []
        cords_y = []
        rgb_array = np.zeros((li,lj,3), 'uint8')
        # image_pred2 = copy.deepcopy(image_pred)
        image_pred2 = image_pred
        for i in range(li):
            print(i)
            for j in range(lj):
                if image_pred[i,j]>60:
                    cont = fill_alg(image_pred2, rgb_array, color, i, j, 0, [], [])
                    color = (color+1)%(len(colores_r))
                    x = sum(cords_x) / float(len(cords_x))
                    y = sum(cords_y) / float(len(cords_y))
                    rgb_array[int(x), int(y), 0] = 0
                    rgb_array[int(x), int(y), 1] = 0
                    rgb_array[int(x), int(y), 2] = 0
        im = Image.fromarray(image_pred.astype(np.uint8))
        im_rgb = Image.fromarray(rgb_array, 'RGB')
        im.save(os.path.join(pred_dir, str(test_id[index]) + '_pred.png'))
        im_rgb.save(os.path.join(pred_dir, str(test_id[index]) + '_predcol.png'))


def predict_web(directorio_entrada, directorio_salida, para_url):
    """
    Funcion que es llamada desde la aplicacion para hacer la segmentacion
    Recibe el directorio donde se encuentran las imagenes,
    el directorio donde las debe dejar y la
    informacion para crear el url que dirija a las imagenes creadas
    """
    start_time = time.time()
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    # Load test data
    cell_segmentation_data = load_test_data(directorio_entrada+"*.png")

    # Preprocess and reshape test data
    x_test = preprocess(cell_segmentation_data[0])
    test_id = cell_segmentation_data[1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # Get model
    model = get_unet()

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    # Load weights
    model.load_weights(weights_path)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    # Make predictions
    start_time_pred = time.time()
    imgs_mask_predict = model.predict(x_test, verbose=1)
    pred_time = time.time()-start_time_pred
    
    print('-' * 30)
    print('Saving predicted masks to files...')
    np.save('imgs_mask_predict.npy', imgs_mask_predict)
    print('-' * 30)
    if not os.path.exists(directorio_salida):
        os.mkdir(directorio_salida)
    # Save predictions as images
    pred_time=pred_time/x_test.shape[0]
    for image_pred, index in zip(imgs_mask_predict, range(x_test.shape[0])):
        start_time = time.time()
        image_pred = image_pred[:, :, 0]
        
        li = image_pred.shape[0]
        lj = image_pred.shape[1]
        image_pred[image_pred > 0.5] *= 255.
        # para colorear la imagen
        color = 0
        cont = 0
        cords_x = []
        cords_y = []
        image_pred3 = copy.deepcopy(image_pred)
        image_pred2 = image_pred
        rgb_array = np.zeros((li,lj,3), 'uint8')
        cell_cont = 1
        Celulas = {'Numero': [], 'Area':[], 'Centroide_x':[], 'Centroide_y':[], 'R':[], 'G': [], 'B': []}
        for i in range(li):
            print(i)
            for j in range(lj):
                if image_pred[i,j]>60:
                    color = (color+1)%(len(colores_r))
                    cords_x = []
                    cords_y = []
                    cont = fill_alg(image_pred2, rgb_array, color, i, j, 0, cords_x, cords_y)
                    print("Area: "+str(cont))
                    x = sum(cords_x) / float(len(cords_x))
                    y = sum(cords_y) / float(len(cords_y))
                    rgb_array[int(x), int(y), 0] = 0
                    rgb_array[int(x), int(y), 1] = 0
                    rgb_array[int(x), int(y), 2] = 0
                    print("Centroide: "+str(x)+" "+str(y))
                    Celulas['Numero'].append(cell_cont)
                    Celulas['Area'].append(cont)
                    Celulas['Centroide_x'].append(x)
                    Celulas['Centroide_y'].append(y)
                    Celulas['R'].append(colores_r[color])
                    Celulas['G'].append(colores_g[color])
                    Celulas['B'].append(colores_b[color])
                    cell_cont+=1
                
        im = Image.fromarray(image_pred3.astype(np.uint8))
        im_rgb = Image.fromarray(rgb_array, 'RGB')
        
        #txt = Image.new('RGB', im_rgb.size, (255,255,255))
        fnt = ImageFont.truetype('/home/j/Desktop/segtracker/fonts/arial.ttf', 16)
        #d = ImageDraw.Draw(txt)
        draw = ImageDraw.Draw(im_rgb)
        for i in range(len(Celulas['Numero'])):
            puntox = Celulas['Centroide_x'][i]
            puntoy = Celulas['Centroide_y'][i]
            numero = Celulas['Numero'][i]
            draw.text((puntoy, puntox), str(numero), "rgb(255,255,255)", font=fnt)
        
        
        df = DataFrame(Celulas, columns= ['Numero', 'Area', 'Centroide_x', 'Centroide_y', 'R', 'G', 'B'])
        export_csv = df.to_csv (os.path.join(directorio_salida, str(test_id[index])
                             + '.csv'), index = None, header=True)
        im.save(os.path.join(directorio_salida, str(test_id[index])
                             + '_pred.png'))
        im_rgb.save(os.path.join(directorio_salida, str(test_id[index])
                             + '_predcol.png'))
        #para mostrar la imagen original hay que quitar el colal final del path
        nuevo_archivo = modelos.Archivo(
            os.path.join(directorio_salida, str(test_id[index]) + '_pred.png'),
            modelos.photos.url(para_url+str(test_id[index]) + '_pred.png'))
        nuevo_archivo_csv = modelos.Archivo(
            os.path.join(directorio_salida, str(test_id[index])
                             + '.csv'),
            modelos.photos.url(para_url+str(test_id[index]) + '.csv'))
        nuevo_archivo_rgb = modelos.Archivo(
            os.path.join(directorio_salida, str(test_id[index]) + '_predcol.png'),
            modelos.photos.url(para_url+str(test_id[index]) + '_predcol.png'))
        modelos.db.session.add(nuevo_archivo)
        modelos.db.session.add(nuevo_archivo_csv)
        modelos.db.session.add(nuevo_archivo_rgb)
        modelos.db.session.commit()
        en_la_sesion = modelos.SesionSalida()
        en_la_sesion.id_sesion = modelos.session['id_sesion']
        en_la_sesion.id_archivo = nuevo_archivo.id_archivo
        en_la_sesion.id_informe = nuevo_archivo_csv.id_archivo
        en_la_sesion.id_gt = nuevo_archivo_rgb.id_archivo
        end_time = (time.time()-start_time)+pred_time
        en_la_sesion.tiempo_ejecucion = end_time
        print(end_time)
        modelos.db.session.add(en_la_sesion)
        modelos.db.session.commit()
        
    return True
    

def predict_web2(directorio_entrada, directorio_salida, test_id):
    """
    Funcion que es llamada desde la aplicacion para hacer la segmentacion
    Recibe el directorio donde se encuentran las imagenes,
    el directorio donde las debe dejar y la
    informacion para crear el url que dirija a las imagenes creadas
    """
    start_time = time.time()
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    # Load test data
    cell_segmentation_data = load_test_data(directorio_entrada+"*.png")

    # Preprocess and reshape test data
    x_test = preprocess(cell_segmentation_data[0])
    # test_id = cell_segmentation_data[1]
    id_dataset=test_id
    test_id =  get_file_names(directorio_entrada+"*.png")

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # Get model
    model = get_unet()

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    # Load weights
    model.load_weights(weights_path)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    # Make predictions
    start_time_pred = time.time()
    imgs_mask_predict = model.predict(x_test, verbose=1)
    pred_time = time.time()-start_time_pred
    
    print('-' * 30)
    print('Saving predicted masks to files...')
    np.save('imgs_mask_predict.npy', imgs_mask_predict)
    print('-' * 30)
    if not os.path.exists(directorio_salida):
        os.mkdir(directorio_salida)
    # Save predictions as images
    pred_time=pred_time/x_test.shape[0]
    for image_pred, index in zip(imgs_mask_predict, range(x_test.shape[0])):
        start_time = time.time()
        image_pred = image_pred[:, :, 0]
        
        #la imagen que se esta procesando
        nombre_archivo = test_id[index]+".png"
        imagen_fuente = Imagen.objects.get(name=nombre_archivo, idDataSet = id_dataset)
        dataset_fuente = DataSet.objects.get(id = id_dataset)
        
        li = image_pred.shape[0]
        lj = image_pred.shape[1]
        image_pred[image_pred > 0.5] *= 255.
        # para colorear la imagen
        color = 0
        cont = 0
        cords_x = []
        cords_y = []
        image_pred3 = copy.deepcopy(image_pred)
        image_pred4 = copy.deepcopy(image_pred)
        image_pred2 = image_pred
        rgb_array = np.zeros((li,lj,3), 'uint8')
        rgb_array2 = np.zeros((li,lj,3), 'uint8')
        cell_cont = 1
        Cell_info = []
        Celulas = {'Numero': [], 'Area':[], 'Centroide_x':[], 'Centroide_y':[], 'R':[], 'G': [], 'B': []}
        for i in range(li):
            print(i)
            for j in range(lj):
                if image_pred4[i,j]>60:
                    color = (color+1)%(len(colores_r))
                    cords_x = []
                    cords_y = []
                    cont = fill_alg(image_pred4, rgb_array2, color, i, j, 0, cords_x, cords_y)
                    print("Area: "+str(cont))
                    x = sum(cords_x) / float(len(cords_x))
                    y = sum(cords_y) / float(len(cords_y))
                    rgb_array[int(x), int(y), 0] = 0
                    rgb_array[int(x), int(y), 1] = 0

                    rgb_array[int(x), int(y), 2] = 0
                    print("Centroide: "+str(x)+" "+str(y))
                    Celulas['Numero'].append(cell_cont)
                    #cont = fill_alg(image_pred2, rgb_array, color, puntox, puntoy, 0, cords_x, cords_y)
                    Celulas['Area'].append(cont)
                    Celulas['Centroide_x'].append(x)
                    Celulas['Centroide_y'].append(y)
                    Celulas['R'].append(colores_r[color])
                    Celulas['G'].append(colores_g[color])
                    Celulas['B'].append(colores_b[color])
                    nx = int(x)
                    ny = int(y)
                    Cell_info.append([(1000*x+y), int(x), int(y)])
                    narea = int(cont)
                    nueva_celula = Celula.objects.create(id_imagen = imagen_fuente, idDataSet = dataset_fuente, x = nx, y = ny, area = narea, num_celula = cell_cont)
                    cell_cont+=1
        Cell_info.sort()      
        im = Image.fromarray(image_pred3.astype(np.uint8))
        #im_rgb = Image.fromarray(rgb_array, 'RGB')
        
        
        #txt = Image.new('RGB', im_rgb.size, (255,255,255))
        fnt = ImageFont.truetype('/home/j/Desktop/segtracker/fonts/arial.ttf', 16)
        #d = ImageDraw.Draw(txt)
        
        color = 0
        for i in range(len(Cell_info)):
            puntox = Cell_info[i][1]
            puntoy = Cell_info[i][2]#Celulas['Centroide_y'][i]
            numero = i+1#Celulas['Numero'][i]
            color = (color+1)%(len(colores_r))
            cords_x = []
            cords_y = []
            rgb_array[int(puntox), int(puntoy), 0] = 0
            rgb_array[int(puntox), int(puntoy), 1] = 0
            cont = fill_alg(image_pred2, rgb_array, color, puntox, puntoy, 0, cords_x, cords_y)
            #draw.text((puntoy, puntox), str(numero), "rgb(255,255,255)", font=fnt)
            
            
        im_rgb = Image.fromarray(rgb_array, 'RGB')
        draw = ImageDraw.Draw(im_rgb)
        for i in range(len(Cell_info)):
            puntox = Cell_info[i][1]
            puntoy = Cell_info[i][2]#Celulas['Centroide_y'][i]
            numero = i+1#Celulas['Numero'][i]
            draw.text((puntoy, puntox), str(numero), "rgb(255,255,255)", font=fnt)
        
        df = DataFrame(Celulas, columns= ['Numero', 'Area', 'Centroide_x', 'Centroide_y', 'R', 'G', 'B'])
        export_csv = df.to_csv (os.path.join(directorio_salida, str(test_id[index])
                             + '.csv'), index = None, header=True)
        im.save(os.path.join(directorio_salida, str(test_id[index])
                             + '_pred.png'))
        im_rgb.save(os.path.join(directorio_salida, str(test_id[index])
                             + '_predcol.png'))
        #para mostrar la imagen original hay que quitar el colal final del path
        nombre_archivo = test_id[index]+".png"
        
        imagen_fuente = Imagen.objects.get(name=nombre_archivo, idDataSet = id_dataset)
        dataset_fuente = DataSet.objects.get(id = id_dataset)
        nuevo_archivo = ImageResult.objects.create(
            id_imagen_base=imagen_fuente, idDataSet=dataset_fuente,dir= directorio_salida,
             name=str(test_id[index]) + '_pred.png')
        nuevo_colores = ColoresResult.objects.create(
            id_imagen_base=imagen_fuente, idDataSet=dataset_fuente,dir= directorio_salida,
             name=str(test_id[index]) + '_predcol.png')
        nuevo_csv = CSVResult.objects.create(
            id_imagen_base=imagen_fuente, idDataSet=dataset_fuente,dir= directorio_salida,
             name=str(test_id[index]) + '.csv')
        """
        #nuevo_csv = CSVResult(
        #    imagen_fuente.get_id(), imagen_fuente.get_id_dataset(), directorio_salida,
        #     str(test_id[index]) + '.csv')
        
        models.db.session.add(nuevo_archivo)
        models.db.session.add(nuevo_csv)
        models.db.session.commit()
        
        nuevo_csv = CSVResult()
        nuevo_csv.id_imagen_base = imagen_fuente.get_id()
        nuevo_csv.idDataSet = imagen_fuente.get_id_dataset()
        nuevo_csv.dir = directorio_salida
        nuevo_csv.name = str(test_id[index]) + '.csv'
        nuevo_archivo.save()
        nuevo_csv.save()
        """
        
        """
        nuevo_archivo = modelos.Archivo(
            os.path.join(directorio_salida, str(test_id[index]) + '_pred.png'),
            modelos.photos.url(para_url+str(test_id[index]) + '_pred.png'))
        nuevo_archivo_csv = modelos.Archivo(
            os.path.join(directorio_salida, str(test_id[index])
                             + '.csv'),
            modelos.photos.url(para_url+str(test_id[index]) + '.csv'))
        nuevo_archivo_rgb = modelos.Archivo(
            os.path.join(directorio_salida, str(test_id[index]) + '_predcol.png'),
            modelos.photos.url(para_url+str(test_id[index]) + '_predcol.png'))
        modelos.db.session.add(nuevo_archivo)
        modelos.db.session.add(nuevo_archivo_csv)
        modelos.db.session.add(nuevo_archivo_rgb)
        modelos.db.session.commit()
        en_la_sesion = modelos.SesionSalida()
        en_la_sesion.id_sesion = modelos.session['id_sesion']
        en_la_sesion.id_archivo = nuevo_archivo.id_archivo
        en_la_sesion.id_informe = nuevo_archivo_csv.id_archivo
        en_la_sesion.id_gt = nuevo_archivo_rgb.id_archivo
        end_time = (time.time()-start_time)+pred_time
        en_la_sesion.tiempo_ejecucion = end_time
        print(end_time)
        modelos.db.session.add(en_la_sesion)
        modelos.db.session.commit()
        """
        
    return True


def predict_web_test(directorio_entrada, directorio_salida):
    """
    Funcion que es llamada desde la aplicacion para hacer la segmentacion
    Recibe el directorio donde se encuentran las imagenes, el
    directorio donde las debe dejar
    Se encarga de interactuar con las pruebas unitarias
    """
    start_time = time.time()
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    # Load test data
    cell_segmentation_data = load_test_data(directorio_entrada+"*.png")

    # Preprocess and reshape test data
    x_test = preprocess(cell_segmentation_data[0])
    test_id = cell_segmentation_data[1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # Get model
    model = get_unet()

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    # Load weights
    model.load_weights(weights_path)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    # Make predictions
    imgs_mask_predict = model.predict(x_test, verbose=1)

    print('-' * 30)
    print('Saving predicted masks to files...')
    np.save('imgs_mask_predict.npy', imgs_mask_predict)
    print('-' * 30)
    if not os.path.exists(directorio_salida):
        os.mkdir(directorio_salida)
    # Save predictions as images
    for image_pred, index in zip(imgs_mask_predict, range(x_test.shape[0])):
        image_pred = image_pred[:, :, 0]
        image_pred[image_pred > 0.5] *= 255.
        im = Image.fromarray(image_pred.astype(np.uint8))
        im.save(os.path.join(directorio_salida, str(test_id[index])
                             + '_pred.png'))
    return True


if __name__ == '__main__':
    predict()
