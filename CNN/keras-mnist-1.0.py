###############################################################################
# Universidade de Passo Fundo
# Prof. Rafael Rieder
#
# Exemplo de construção de uma rede neural de aprendizado profundo densamente
# conectada (FC-ANN) usando Tensorflow/ Keras
#
# Dataset: MNIST (dígitos manuscritos de 0 a 9, grayscale)
# http://yann.lecun.com/exdb/mnist/
###############################################################################

# Importamos o conjunto de dados MNIST
from keras.datasets import mnist

# Para visualizarmos bem a sequência de camadas do modelo 
# vamos usar o modulo do Keras chamado Sequential 
# (https://keras.io/getting-started/sequential-model-guide/)
from keras.models import Sequential

# Como o exemplo é de um modelo simples, vamos utilizar
# camadas densas, as quais são camadas onde cada unidade
# ou neurônio estará conectado a cada neurônio na próxima camada
# Fully-connected net
from keras.layers import Dense

# Módulo do Keras responsável por várias rotinas de pré-processamento 
# (https://keras.io/utils/).
from keras.utils import np_utils

# Aqui estamos carregando o conjunto de dados em subconjuntos de 
# treinamento e teste.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]

# Com o intuito de amenizar o uso de memória podemos atribuir um nível de precisao
# dos valores de pixel com sendo 32 bits (float32)
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Podemos normalizar os valores de pixels para o intervalo 0 e 1 
# dividindo cada valor pelo máximo de 255, visto que os valores 
# de pixel no dataset estão em escala de cinza entre 0 e 255.
X_train = X_train / 255
X_test = X_test / 255

# Como estamos trabalhando com um problema de classificação
# multiclasses, pois temos vários tipos de dígitos, vamos 
# represantá-los em categorias usando a metodologia de 
# one-hot-encoding aqui representada pela funcao to_categorical.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Número de tipos de digitos encontrados no MNIST (no caso são 10 classes, de 0 a 9)
num_classes = y_test.shape[1]

# Modelo básico de uma camada onde inicializamos um modelo sequêncial
# com suas funções de ativação, e o compilamos usando gradiente descendente (sgd) e
# acurácia como métrica de avaliação.

def base_model():
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
	model.add(Dense(64, activation="relu"))
	model.add(Dense(num_classes, activation='softmax', name='preds'))
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return model

model = base_model()

# O método summary revela quais são as camadas
# que formam o modelo, seus formatos e o número
# de parâmetros envolvidos em cada etapa.
model.summary()

# Processo de treinamento do modelo
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=100)

# Avaliação da performance do modelo recém treinado
scores = model.evaluate(X_test, y_test, verbose=0)
print("\nProbabilidade de acerto: %.2f%%" % (scores[1]*100))
print("Probabilidade de erro..: %.2f%%" % (100-scores[1]*100))