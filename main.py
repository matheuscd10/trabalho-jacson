from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

# Diretório raiz das imagens
data_dir = 'C:/visaocomputacional/aula-git/cavalo/horse-or-human'

# Pré-processamento das imagens
image_size = (32, 32)  # Defina o tamanho desejado para as imagens
batch_size = 32  # Defina o tamanho do lote (batch size)
num_classes = 2  # Número de classes: cavalo e humano

# Criar gerador de dados para treinamento
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalização dos valores dos pixels (0-255 para 0-1)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'  # As classes são binárias (horse ou human), então utilize class_mode='binary'
)

# Criação do modelo sequencial
model = Sequential()

# Adicionar camadas convolucionais, de pooling e densas ao modelo
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compilação do modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

num_epochs = 10  # Defina o número de épocas desejado

# Treinamento do modelo
model.fit(train_generator, epochs=num_epochs)

# Avaliação do modelo
test_loss, test_accuracy = model.evaluate(train_generator)

"""abrir a imagem"""

# Carregar a imagem
image = Image.open('C:/visaocomputacional/aula-git/cavalo/horse-or-human/horses/teste.jpg')

# Pré-processamento da imagem
image = image.resize(image_size)
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

class_names = {
    0: "cavalo",
    1: "humano"
}

# Fazer a previsão
prediction = model.predict(image)
predicted_class = np.argmax(prediction)
predicted_class_name = class_names[predicted_class]

# Imprimir a classe prevista
print('Classe prevista:', predicted_class_name)