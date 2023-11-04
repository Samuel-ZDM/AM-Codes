###############################################################################
# Universidade de Passo Fundo
# Prof. Rafael Rieder
#
# Exemplo de construção de uma rede neural convolucional de aprendizado profundo
# (CNN) usando Tensorflow/ Keras
#
# Dataset: MNIST (dígitos manuscritos de 0 a 9, grayscale)
# http://yann.lecun.com/exdb/mnist/
###############################################################################

# Importação de dependências 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# Como a organização do dataset é (channels, rows, cols), então deve-se usar "channels_first"
K.set_image_data_format('channels_first')

# Gerando nossos subconjuntos de treinamento e teste
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Como estamos trabalhando em escala de cinza, podemos
# definir a dimensão do pixel como sendo 1.
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Normalizamos nossos dados de acordo com variação da escala de cinza
X_train = X_train / 255
X_test = X_test / 255

# Aplicamos a solução de one-hot-encoding para classificação multiclasses.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Número de tipos de dígitos encontrados no MNIST (de 0 a 9)
num_classes = y_test.shape[1]

def deeper_cnn_model():
    model = Sequential()

    # A Convolution2D será a nossa camada de entrada. Podemos observar que ela possui 
    # 60 mapas de features com tamanho de 5 × 5 e 'relu' como funcao de ativacao.
    # (se a imagem manipulada fosse colorida, seria Conv3D) 
    model.add(Conv2D(60, (5, 5), input_shape=(1, 28, 28), activation='relu'))

    # A camada MaxPooling2D será nossa segunda camada, onde teremos um amostragem de 
    # dimensões 2 x 2.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Duas novas camadas convolucionais com 30 mapas de features cada, dimensões 3 × 3 
    # e 'relu' como funcao de ativacao. 
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    # Dropout com probabilidade de 30% (percentual de neurônios que terão a entrada descartada)
    model.add(Dropout(0.3))

    # Flatten preparando os dados para a camada fully connected. 
    model.add(Flatten())

    # Camada fully connected de 512 neurônios conectados com a camada flatten.
    model.add(Dense(512, activation='relu'))

    # A camada de saída possui o número de neurônios compatível com o 
    # número de classes a serem classificadas, com uma função de ativação
    # do tipo 'softmax'. Todos esses nodos estão conectados com todos os neurônios da camada anterior.
    model.add(Dense(10, activation='softmax'))

    # Compilação do modelo usando otimizador adam
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Gera modelo
model = deeper_cnn_model()

# O método summary revela quais são as camadas
# que formam o modelo, seus formatos e o número
# de parâmetros envolvidos em cada etapa.
model.summary()

# Processo de treinamento do modelo. 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=100)

# Avaliação da performance do modelo CNN
scores = model.evaluate(X_test, y_test, verbose=0)
print("\nProbabilidade de acerto: %.2f%%" % (scores[1]*100))
print("Probabilidade de erro..: %.2f%%" % (100-scores[1]*100))