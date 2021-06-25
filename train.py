import argparse
import os
import pathlib
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Kerasa / TensorFlow
from loss import depth_loss_function
from utils import load_test_data
from model import create_model
from data import get_nyu_train_test_data
from callbacks import get_nyu_callbacks

from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import os

# Argument Parser
parser = argparse.ArgumentParser(description='Encoder-decoder Based Monocular Depth Estimation')
parser.add_argument('--data', default='nyu', type=str, help='Training Dataset.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--bs', type=int, default=4, help='Batch Size')
parser.add_argument('--epochs', type=int, default=20, help='Number of Epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to Use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to Use')
parser.add_argument('--mindepth', type=float, default=10.0, help='Minimum of Input Depths')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of Input Depths')
parser.add_argument('--name', type=str, default='depth_nyu', help='A Name to Attach to the Training Session')
parser.add_argument('--checkpoint', type=str, default='', help='Start Training from an Existing Model.')
parser.add_argument('--full', dest='full', action='store_true',
                    help='Full Training with Metrics, Checkpoints, and Image Samples.')

args = parser.parse_args()

# Inform about multi-gpu training
if args.gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('Will use GPU ' + args.gpuids)
else:
    print('Will use ' + str(args.gpus) + ' GPUs.')

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# config = tf.ConfigProto()
# config.allow_soft_placement = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# Create the model
model = create_model(existing=args.checkpoint)

# Data loaders
if args.data == 'nyu': train_generator, test_generator = get_nyu_train_test_data(args.bs)

# Training session details
runID = str(int(time.time())) + '-n' + str(len(train_generator)) + '-e' + str(args.epochs) + '-bs' + str(
    args.bs) + '-lr' + str(args.lr) + '-' + args.name
outputPath = './models/'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

# Multi-gpu setup:
basemodel = model
if args.gpus > 1: model = multi_gpu_model(model, gpus=args.gpus)

# Optimizer
optimizer = Adam(lr=args.lr, amsgrad=True)

# Compile the model
print('\n\n\n', 'Compiling model..', runID, '\n\n\tGPU ' + (str(args.gpus) + ' gpus' if args.gpus > 1 else args.gpuids)
      + '\t\tBatch size [ ' + str(args.bs) + ' ] ' + ' \n\n')
model.compile(loss=depth_loss_function, optimizer=optimizer)

print('Ready for training!\n')

# Callbacks
callbacks = []
if args.data == 'nyu': callbacks = get_nyu_callbacks(model, basemodel, train_generator, test_generator,
                                                     load_test_data() if args.full else None, runPath)

# Start time
start_time = time.time()

# Start training
model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=args.epochs,
                    shuffle=True)

end_time = time.time()

with open("timing.txt", "w") as f:
    f.write("Training time: {} s".format(end_time - start_time))

# Save the final trained model:
basemodel.save(runPath + '/model.h5')
