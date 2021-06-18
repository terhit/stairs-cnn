import tensorflow as tf
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',
                        type=str,
                        help='Path to model to be converted, e.g. results/model_name/model')
    args = parser.parse_known_args()[0]
    return args

args = parse_arguments()
converter = tf.lite.TFLiteConverter.from_saved_model(args.model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_quantized_model)