import tensorflow as tf
import pathlib
import argparse
import os
import numpy as np

celbi_class = ['biomassa', 'cepos', 'estilha', 'molhos', 'troncos','troncos com casca']

def find_class(dataset):
  labels =  [0, 0, 0, 0, 0 ,0]

  img_height = 180
  img_width = 180

  for image in dataset:
    path = pathlib.Path(image)

    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    interpreter.set_tensor(input_details[0]['index'], img_array)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index']) # outputs the probability of being each label (i.e biomassa, estilha, etc)

    # print(output_data[0])
    print(output_data)

    max_index = output_data[0].argmax() # find the most probable label

    score = tf.nn.softmax(output_data[0])
    print(score)

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(celbi_class[np.argmax(score)], 100 * np.max(score))
    )

    print(image)
    print(max_index)

    labels[max_index] +=1 # increase matches counter


  max_index = labels.index(max(labels)) # get index of label with more matches

  return max_index



# Argument parser
parser = argparse.ArgumentParser(description='Correct Micasense reflectance images. Update EXIF information.')
#  parser.add_argument("-b", "--base", type=str, default='/Users/jpc/Documents/PDMFC/data', help='Base directory')
parser.add_argument("-b", "--base", type=str, default='/code/data', help='Base directory')
parser.add_argument("-d", "--dir", type=str, default='data_raw', help='File directory')
parser.add_argument("-u", "--uid", type=str, default='1', help='User ID')
args = parser.parse_args()



#setup
TF_MODEL_FILE_PATH = 'tensorflowModels/model.tflite' # The default path to the saved TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
interpreter.allocate_tensors()


data_dir = os.path.join(args.base, args.uid, args.dir, 'Mission2')
data_dir = pathlib.Path(data_dir).with_suffix('')
print(data_dir)

dataset = list(data_dir.glob('*'))
print(celbi_class[find_class(dataset)])