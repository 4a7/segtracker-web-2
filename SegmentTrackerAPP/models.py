from django.db import models
import datetime
import math

class Usuario(models.Model):
    id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length = 100, default="John")
    apellido = models.CharField(max_length = 100, default="Doe")
    correo = models.CharField(max_length = 100, default="jdoe@gmail.com")
    institucion = models.CharField(max_length = 100, default="Doe Institute")
    password = models.CharField(max_length = 256, default="123456")

    def __str__(self):
        return self.nombre
    def getId(self):
        return self.id   

class DataSet(models.Model):
    id = models.AutoField(primary_key=True)
    id_user = models.IntegerField(default=1)
    date = models.DateField('Date', default=datetime.date.today())
    name = models.CharField(max_length = 100, default="dataSetName")
    desc = models.CharField(max_length = 1024, default="dataSetDesc")
    def __date__(self):
        return self.date

class Imagen(models.Model):
    id = models.AutoField(primary_key=True)
    idDataSet = models.IntegerField(default=1)
    dir = models.CharField(max_length = 256, default="dirImage")
    name = models.CharField(max_length = 256, default="nameImage")
    
    def get_id_dataset(self):
        return self.idDataSet
    def get_id(self):
        return self.id
    
class ImageResult(models.Model):
    id = models.AutoField(primary_key=True)
    id_imagen_base = models.ForeignKey(Imagen, on_delete=models.DO_NOTHING)
    idDataSet = models.ForeignKey(DataSet, on_delete=models.DO_NOTHING)
    dir = models.CharField(max_length = 256, default="dirImage")
    name = models.CharField(max_length = 256, default="nameImage")
    
    def get_name(self):
        return str(self.name)
    def get_imagen_base(self):
        return self.id_imagen_base
    def get_full_path(self):
        return self.dir+self.name
    
class ColoresResult(models.Model):
    id = models.AutoField(primary_key=True)
    id_imagen_base = models.ForeignKey(Imagen, on_delete=models.DO_NOTHING)
    idDataSet = models.ForeignKey(DataSet, on_delete=models.DO_NOTHING)
    dir = models.CharField(max_length = 256, default="dirImage")
    name = models.CharField(max_length = 256, default="nameImage")
    
    def get_name(self):
        return self.name
    def get_imagen_base(self):
        return self.id_imagen_base
    
    
class CSVResult(models.Model):
    id = models.AutoField(primary_key=True)
    id_imagen_base = models.ForeignKey(Imagen, on_delete=models.DO_NOTHING)
    idDataSet = models.ForeignKey(DataSet, on_delete=models.DO_NOTHING)
    dir = models.CharField(max_length = 256, default="dirImage")
    name = models.CharField(max_length = 256, default="nameImage")
    
    def get_full_path(self):
        return self.dir+self.name
    
class Celula(models.Model):
    id = models.AutoField(primary_key=True)
    id_imagen = models.ForeignKey(Imagen, on_delete=models.DO_NOTHING)
    idDataSet = models.ForeignKey(DataSet, on_delete=models.DO_NOTHING)
    x = models.IntegerField(default=1)
    y = models.IntegerField(default=1)
    area = models.IntegerField(default=1)
    num_celula = models.IntegerField(default=1)
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def distancia(self, tx, ty):
        return math.sqrt(float((tx-self.x)**2+(ty-self.y)**2))
    def get_area(self):
        return self.area
        
class ImagenRastreo(models.Model):
    id = models.AutoField(primary_key=True)
    idDataSet = models.ForeignKey(DataSet, on_delete=models.DO_NOTHING)
    dir = models.CharField(max_length = 256, default="dirImage")
    name = models.CharField(max_length = 256, default="nameImage")
    num_secuencia = models.IntegerField(default=1)
    num_celula = models.IntegerField(default=1)
    
    def get_name(self):
        return self.name
    def get_imagen_base(self):
        return self.id_imagen_base
    def get_num_celula(self):
        return self.num_celula
    
