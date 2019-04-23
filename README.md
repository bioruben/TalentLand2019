# Computer Vision and Artificial Intelligence for robots by Carbon Robotics

Material utilizado dentro del workshop presentado en Talent Land 2019 por Rubén Alvarez

## Instalar dependencias necesarias

Despues de clonar este repositorio, es necesario instalar:

- Python3
- OpenCV
- Tensorflow
- Keras

## Modelos

En el siguiente link puede bajar los modelos entrenados utilizados en los siguientes ejercicios.

```https://bit.ly/2IDWDgR```


### Aplicaciones

Hay 3 códigos diferentes:

- object_detection.py: Se encarga de hacer detección de objetos utilizando el modelo de YOLOv3 y guarda el resultado en un video en el mismo folder.

- test_cnn.py: corre un detector para clasificar imágenes entre Luke Skywalker o Darth Vader.

- training_cnn.py: Entrena la CNN utilizada en test_cnn.py, aquí se puede modificar la arquitectura de la CNN así como también si se desea clasificar dos clases diferentes.



## Usage

Todos los códigos tienen un menú de help para poderlo correr, ejemplo:

```python3 test_cnn.py -h```

Si se desea probar la CNN después de haber bajado los pesos del modelo o haberlo entrenado cada usuario.

```python3 test_cnn.py -i /path/to/the/image/folder```