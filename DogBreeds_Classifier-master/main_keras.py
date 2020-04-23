import os
import numpy as np
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import keras
from dataset import Dataset
from cnn_base import CNN_basemodels
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %%$################## GLOBAL VAR ########################
TRAIN_PATH = '/media/data/MOA144/Assignment 4/train'
VAL_PATH = '/media/data/MOA144/Assignment 4/val'
TEST_PATH = '/media/data/MOA144/Assignment 4/test'
TEST_LABELS = '/media/data/MOA144/Assignment 4/MO444_dogs_test.txt'
IMG_HEIGHT = 299
IMG_WIDTH = 299
IMG_CHANNELS = 3
OUTPUT_SHAPE = 83
FINETUNE = True
NN_NAME = ['InceptionResNetV2_FT91pt8',
           'InceptionResNetV2_StrongerAug_FT93pt3',
           'Xception_FT92pt8',
           'InceptionV3_StrongerAug_FT91pt2']
USE_TEST = True

# %%################## HYPERPARAMETERS ######################
LEARN_RATE_0 = 0.001
LEARN_RATE_FINETUNE = 0.00001
N_EPOCH = 20
MB_SIZE = 10
AUGMENTATION = dict(
    rotation_range=360,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.3,
    channel_shift_range=0,
    horizontal_flip=True,
    vertical_flip=True,
)
# %%################## DATASETS ############################
if not USE_TEST:
    train = Dataset(TRAIN_PATH,
                    shuffle=True,
                    batch_size=MB_SIZE,
                    tgt_size=(IMG_HEIGHT, IMG_WIDTH),
                    color=True,
                    augmentation=AUGMENTATION)

    val = Dataset(VAL_PATH,
                  shuffle=True,
                  batch_size=MB_SIZE,
                  tgt_size=(IMG_HEIGHT, IMG_WIDTH),
                  color=True,
                  augmentation=None,
                  #              len_mod=0.2
                  )

else:
    train = Dataset([TRAIN_PATH, VAL_PATH],
                    shuffle=True,
                    batch_size=MB_SIZE,
                    tgt_size=(IMG_HEIGHT, IMG_WIDTH),
                    color=True,
                    augmentation=AUGMENTATION)

    val = Dataset(TEST_PATH,
                  shuffle=True,
                  batch_size=MB_SIZE,
                  tgt_size=(IMG_HEIGHT, IMG_WIDTH),
                  color=True,
                  augmentation=None,
                  load_labels_from_file=TEST_LABELS)

x_trn = train.x
y_trn = train.y
x_vld = val.x
y_vld = val.y


# %%################# KERAS MODEL ##########################
nets = []
if not FINETUNE:
    nets.append(CNN_basemodels(InceptionV3, OUTPUT_SHAPE,
                               lr=LEARN_RATE_0, training_regime='top',
                               name=NN_NAME,
                               top=[Dropout(0.5)]
                               ))

else:
    if type(NN_NAME) is list:
        for name in NN_NAME:
            nets.append(CNN_basemodels.load_model(name))
    else:
        nets.append(CNN_basemodels.load_model(NN_NAME))
    for net in nets:
        net.set_lr(LEARN_RATE_FINETUNE)
        net.set_trainable_layers('all')

# %% ############# TRAINING ##########################
cb = EarlyStopping(monitor='val_loss', min_delta=0,
                   patience=2, verbose=0, mode='auto')

mc = ModelCheckpoint(NN_NAME, monitor='val_loss', save_best_only=True)

train_args = dict(
    generator=train,
    steps_per_epoch=len(y_trn) // MB_SIZE,
    epochs=N_EPOCH,
    callbacks=[cb, mc],
    validation_data=val,
    validation_steps=len(y_vld) // MB_SIZE,
    workers=4, use_multiprocessing=True)

for net in nets:
    print('Starting training: ', net.name)

    net.fit_generator(**train_args)

if not FINETUNE:
    for net in nets:
        net.set_lr(LEARN_RATE_FINETUNE)
        net.set_trainable_layers('all')

        print('Starting training: ', net.name)

        net.fit_generator(**train_args)

# %%################## ENSAMBLE AND TTA ###################
TTA_LIST = [None,
            dict(horizontal_flip=True),
            #            dict(vertical_flip=True),
            dict(rotation=90),
            dict(rotation=180),
            dict(rotation=270),
            ]

pred_trn, y_trn = train.TTA(
    nets, batch_size=MB_SIZE, augmentations=TTA_LIST, len_reducer=0.3)
pred_vld, y_vld = val.TTA(nets, batch_size=MB_SIZE, augmentations=TTA_LIST)

# Soft voting
pred = np.sum(pred_vld, axis=(2, 3))
pred = np.argmax(pred, axis=-1)
print('Val acc: ', np.mean(np.equal(np.argmax(y_vld, axis=-1), pred).astype(np.uint8)))

pred_trn = np.reshape(pred_trn, [pred_trn.shape[0], -1])
y_trn = np.argmax(y_trn, axis=-1)
pred_vld = np.reshape(pred_vld, [pred_vld.shape[0], -1])
y_vld = np.argmax(y_vld, axis=-1)

# ########################### LR #############################
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=10)
LR.fit(pred_trn, y_trn)
LR.score(pred_vld, y_vld)

#%% ####################### LightGBM ##########################
import lightgbm as lgb

d_train = lgb.Dataset(pred_trn, label=y_trn)
lgb_eval = lgb.Dataset(pred_vld, y_vld, reference=d_train)

params = {}
params['learning_rate'] = 0.1
params['boosting_type'] = 'gbdt'
params['objective'] = 'multiclass'
params['metric'] = 'multi_logloss'
params['num_class'] = OUTPUT_SHAPE
params['sub_feature'] = 0.5
params['num_leaves'] = 300
#params['min_data'] = 50
params['max_depth'] = 100
#params['device'] = 'gpu'

clf = lgb.train(params, d_train, 100, valid_sets=lgb_eval)
y_pred = clf.predict(pred_vld, num_iteration=clf.best_iteration)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(np.argmax(y_pred, axis=-1), y_vld)
print(accuracy)

#%% ################## SVM ###################################
from sklearn import svm
clf = svm.SVC(C=1.0, cache_size=500, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', degree=3, gamma='auto', kernel='linear',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
clf.fit(pred_trn, y_trn)
print(clf.score(pred_vld, y_vld))

# %%############## F1 Score #################################
from sklearn.metrics import f1_score, confusion_matrix
print(f1_score(y_vld, pred, average='weighted'))
conf_matrix = confusion_matrix(y_vld, pred)
