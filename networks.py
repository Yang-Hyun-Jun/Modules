import os
import threading
import numpy as np

if os.environ["KERAS_BACKEND"] == "tensorflow":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, Flatten
    from tensorflow.keras.layers import BatchNormalization, Dropout, MaxPooling2D
    from tensorflow.keras.optimizers import SGD

elif os.environ["KERAS_BACKEND"] == "plaidml.keras.backend":
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Conv2D, Flatten
    from keras.layers import BatchNormalization, Dropout, MaxPooling2D
    from keras.optimizers import SGD

""" 신경망(Network) 클래스 """
class Network:
    lock = threading.Lock() #클래스 변수

    def __init__(self, input_dim=0, output_dim=0, lr=0.001,
                 shared_network=None, activation="sigmoid", loss="mse"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss

    def predict(self, sample):
        with self.lock: #스레드 간 간섭 방지 클래스 변수 호출, with 문으로 자동 열고 닫기
            return self.model.predict(sample).flatten()

    def train_on_batch(self, x, y):
        loss = 0.0
        with self.lock:
            loss = self.model.train_on_batch(x, y)
        return loss

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

    #정적 메소드 @staticmethod: 객체 필드와 독립적이지만 로직상 클래스내에 포함되는 메소드에 사용된다.
    #클래스 메소드 @classmethod: 정적 메소드지만 cls를 통해 인스턴스 변수에 접근 가능
    #신경망 종류에 따라 공유 신경망을 획득하는 클래스 메소
    @classmethod
    def get_shared_network(cls, net="dnn", num_steps=1, input_dim=0):
        if net == "dnn":
            return DNN.get_network_head(Input((input_dim,)))
        elif net == "lstm":
            return LSTMNetwork.get_network_head(Input((num_steps, input_dim)))
        elif net == "cnn":
            return CNN.get_network_head(Input((1, num_steps, input_dim)))

""" DNN """
class DNN(Network): #Network 클래스 상속
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        inp = None
        output = None

        #공유신경망 없으면 직접 생성한 DNN 아키텍쳐(get_network_head)에서 input, ouput 가져오기
        if self.shared_network is None:
            inp = Input((self.input_dim,))
            output = self.get_network_head(inp).output
        #공유신경망 있으면 있는 걸로 input, ouput 가져오기
        else:
            inp = self.shared_network.input
            output = self.shared_network.output

        output = Dense(self.output_dim, activation=self.activation,
                       kernel_initializer="random_normal")(output) #최종 출력 노드
        self.model = Model(inp, output) #inp, output 연결
        self.model.compile(optimizer=SGD(learning_rate=self.lr), loss=self.loss) #컴파일

    @staticmethod #정적 메소드
    def get_network_head(inp):
        output = Dense(256, activation="sigmoid", kernel_initializer="random_normal")(inp)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)

        output = Dense(128, activation="sigmoid", kernel_initializer="random_normal")(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)

        output = Dense(64, activation="sigmoid", kernel_initializer="random_normal")(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)

        output = Dense(32, activation="sigmoid", kernel_initializer="random_normal")(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim)) #(배치개, input차원)으로 reshape
        return super().train_on_batch(x ,y) #메소드 오버라이딩

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample) #메소드 오버라이딩

""" LSTM """
class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps

        inp = None
        output = None

        if self.shared_network is None:
            inp = Input((self.num_steps, self.input_dim))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output

        output = Dense(self.output_dim, activation=self.activation,
                       kernel_initializer="random_normal")(output)
        self.model = Model(inp, output)
        self.model.compile(optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = LSTM(256, dropout=0.1, return_sequences=True, stateful=False,
                      kernel_initializer="random_normal")(inp)
        output = BatchNormalization()(output)

        output = LSTM(128, dropout=0.1, return_sequences=True, stateful=False,
                      kernel_initializer="random_normal")(inp)
        output = BatchNormalization()(output)

        output = LSTM(64, dropout=0.1, return_sequences=True, stateful=False,
                      kernel_initializer="random_normal")(inp)
        output = BatchNormalization()(output)

        output = LSTM(32, dropout=0.1, return_sequences=True, stateful=False,
                      kernel_initializer="random_normal")(inp)
        output = BatchNormalization()(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)

""" CNN """
class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps

        inp = None
        output = None

        if self.shared_network is None:
            inp = Input((self.num_steps, self.input_dim, 1))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output

        output = Dense(self.output_dim, activation=self.activation,
                       kernel_initializer="random_normal")(output)
        self.model = Model(inp, output)
        self.model.compile(optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = Conv2D(256, kernel_size=(1,5), padding="same", activation="sigmoid",
                        kernel_initializer="random_normal")(inp)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)

        output = Conv2D(128, kernel_size=(1,5), padding="same", activation="sigmoid",
                        kernel_initializer="random_normal")(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)

        output = Conv2D(64, kernel_size=(1,5), padding="same", activation="sigmoid",
                        kernel_initializer="random_normal")(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)

        output = Conv2D(32, kernel_size=(1,5), padding="same", activation="sigmoid",
                        kernel_initializer="random_normal")(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)

        output = Flatten()(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().predict(sample)




















