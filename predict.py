import numpy as np
import os, os.path
import argparse
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                    type=str,
                    default='data',
                    help='Folder path to extracted frames')
    parser.add_argument('--model-dir',
                        type=str,
                        default='model',
                        help='Model dir, e.g. model')
    parser.add_argument('--img-width',
                        type=int,
                        default=300)
    parser.add_argument('--img-height',
                        type=int,
                        default=300)
    parser.add_argument('--batch-size',
                        type=int,
                        default=124,
                        help='Batch size')
    args = parser.parse_known_args()[0]
    return args

def load_data(img_width, img_height, test_data_dir, batch_size):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=args.batch_size,
        class_mode='binary',
        shuffle=False)
    
    return test_generator

def allocate_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'], (124, 300, 300, 3))
    interpreter.resize_tensor_input(output_details[0]['index'], (124, 1))
    interpreter.allocate_tensors()

    return interpreter, input_details, output_details

def predict(model_path, test_generator):
    interpreter, input_details, output_details = allocate_interpreter(model_path)

    val_image_batch, _ = next(iter(test_generator))

    interpreter.set_tensor(input_details[0]['index'], val_image_batch)
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    y_hat = np.round(np.squeeze(interpreter.get_tensor(output_details['index'])))

    return y_hat

args = parse_arguments()

test_data_dir = os.path.join(args.data_dir, 'test')
test_generator = load_data(
                    args.img_width,
                    args.img_height,
                    test_data_dir,
                    args.batch_size)

model_path = os.path.join(args.model_dir, 'model')

y_val = test_generator.classes
y_hat = predict(model_path, test_generator)

acc = accuracy_score(y_val, y_hat)
precision = precision_score(y_val, y_hat)
recall = recall_score(y_val, y_hat)
f1 = f1_score(y_val, y_hat)

print('acc: %s \nprecision: %s \nrecall: %s \nf1: %s' % (acc, precision, recall, f1))

with open('test_score.txt', 'w') as f:
    f.write('arguments: ' + str(args) + '\nacc: ' + str(acc) + '\nprecision: ' + str(precision) + 
    '\nrecall: ' + str(recall) + '\nf1: ' + str(f1))
    f.close()