import os, os.path
import argparse
from datetime import date, datetime
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.applications.inception_v3 import InceptionV3

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        type=str,
                        default='data',
                        help='Folder path for extracted frames')
    parser.add_argument('--model-name',
                        type=str,
                        default='InceptionV3',
                        help='Model name')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Epochs')
    parser.add_argument('--lr',
                        type=float,
                        default=0.00005,
                        help='Learning rate')
    parser.add_argument('--model-optimizer',
                        type=str,
                        default='adam',
                        help='Model optimizer, e.g. adam, sgd')
    parser.add_argument('--pooling',
                        type=str,
                        default='min',
                        help='Pooling function, e.g. min, max, avg')
    parser.add_argument('--l2',
                        type=float,
                        default=0.01,
                        help='L2-regularizer')
    parser.add_argument('--img-width',
                        type=int,
                        default=300)
    parser.add_argument('--img-height',
                        type=int,
                        default=300)
    parser.add_argument('--dropout',
                        type=float,
                        default=0.5)
    parser.add_argument('--early-stopping',
                        type=float,
                        default=5)
    args = parser.parse_known_args()[0]
    return args

def select_optimizer(opt, lr):
    if(opt=='sgd'):
        return optimizers.SGD(lr=lr)
    elif(opt=='adagrad'):
        return optimizers.Adagrad(lr=lr)
    elif(opt=='RMSprop'):
        return optimizers.RMSprop(lr=lr)
    else:
        return optimizers.Adam(lr=lr)

def create_base_model(img_width, img_height, pooling):
    return InceptionV3(weights='imagenet', 
        input_shape=(img_width, img_height, 3), 
        include_top=False,
        pooling=pooling)

def define_model(img_width, img_height, opt, lr, pooling, dropout, l2_param):
    # General features are learned from a Inception V3 base model 
    # pre-trained with ImageNet dataset
    base_model = create_base_model(img_width, img_height, pooling)
    base_model.trainable = False
    optimizer = select_optimizer(opt, lr)

    # Task specific features are learned on the classification layers
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(l2_param)))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(l2_param)))
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(l2_param)))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def load_data(img_width, img_height, train_data_dir, validation_data_dir, batch_size):
    # As the training dataset is limited, several data augmentation methods are applied
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.1,
        horizontal_flip=True
        )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator, validation_generator

args = parse_arguments()

date = datetime.now().strftime("%Y%m%d-%H%M")
MODEL_FILE = date + "_" + args.model_name

train_data_dir = os.path.join(args.data_dir, 'train')
validation_data_dir = os.path.join(args.data_dir, 'validation')

log_dir = os.path.join('results', MODEL_FILE, 'tensorboard')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
callback = EarlyStopping(monitor='loss', patience=args.early_stopping)

model = define_model(
            args.img_width, 
            args.img_height, 
            args.model_optimizer, 
            args.lr, 
            args.pooling, 
            args.dropout, 
            args.l2)

train_generator, validation_generator = load_data(
                                            args.img_width, 
                                            args.img_height, 
                                            train_data_dir, 
                                            validation_data_dir,
                                            args.batch_size)

history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator.filenames) // args.batch_size,
            epochs=args.epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator.filenames) // args.batch_size,
            callbacks=[tensorboard_callback, callback])

model.save(os.path.join('results', MODEL_FILE, 'model'))
