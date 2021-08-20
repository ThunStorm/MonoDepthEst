import argparse
import glob
import os

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='Encoder-decoder Based Monocular Depth Estimation')
parser.add_argument('--model', default='./trained models/model.h5', type=str, help='Trained Keras model.')
parser.add_argument('--input', default='pics/*.jpg', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images(glob.glob(args.input))
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

# matplotlib problem on ubuntu terminal fix
# matplotlib.use('TkAgg')

# Display results
# viz = display_images(outputs.copy(), inputs.copy())
# plt.figure(figsize=(10, 5))
viz = display_images(outputs.copy())
plt.figure(figsize=(5, 5))
plt.imshow(viz)
plt.savefig(('maps/dm_' + args.input[-5:]).replace("*", "combined"))
plt.show()
