import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam


class CNN_basemodels():
    load = False

    def __init__(self, Keras_Model, classes, lr=0.001, training_regime='top', top=[], name=None):
        if not self.load:
            self.Keras_Model = Keras_Model
            self.classes = classes
            self.top = top
            self.name = name
            self.history = []

            self.optmizer = Adam(lr=lr)

            self.create_graph()
            self.set_trainable_layers(training_regime)

    def create_graph(self):
        self.base_model = self.Keras_Model(
            weights='imagenet', include_top=False)
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        if len(self.top) > 0:
            for layer in self.top:
                x = layer(x)
        predictions = Dense(self.classes, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.input, outputs=predictions)

    def compile(self):
        self.model.compile(optimizer=self.optmizer, loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def set_lr(self, lr):
        self.optmizer = Adam(lr=lr)
        self.compile()

    def set_trainable_layers(self, n):
        ''' Required to fix weird behavior of reinitializing weights if
        self.model is used'''
        if self.load:
            m = self.model
        else:
            m = self.base_model

        if n == 'all':
            for layer in m.layers:
                layer.trainable = True
        elif n == 'top':
            for layer in m.layers:
                layer.trainable = False
        else:
            for layer in m.layers[:n]:
                layer.trainable = False
            for layer in m.layers[n:]:
                layer.trainable = True

        self.compile()

    def fit_generator(self, **kwargs):
        self.history.append(self.model.fit_generator(**kwargs))

    def evaluate_generator(self, **kwargs):
        self.model.evaluate_generator(**kwargs)

    def predict(self, **kwargs):
        self.model.predict(**kwargs)

    def save_weights(self, path):
        self.model.save_weights(path)

    def save_model(self, path):
        self.model.save(path)

    @classmethod
    def load_model(cls, path):
        cls.load = True
        instance = cls(None, None)
        instance.name = path
        instance.model = load_model(path)
        return instance
