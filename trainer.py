
import time
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import Tensorboard

LOAD_TRAIN_FILES = False
LOAD_PREV_MODEL = False

TRAINING_CHUNK_SIZE = 50
PREV_MODEL_NAME = ""
VALIDATION_GAME_COUNT = 50
NAME = "phase1-{}".format(int(time.time()))
EPOCHS = 1

TRAINING_DATA_DIR = "games"

training_file_names = []

for f in os.listdir(TRAINING_DATA_DIR):
    training_file_name.append(os.path.join(TRAINING_DATA_DIR, f))

print("After the threshold we have {} fames.".format(len(training_file_names)))

random.shuffle(training_file_names)

if LOAD_TRAIN_FILES:
    test_x = np.load("test_x.npy")
    test_y = np.load("test_y.npy")
else:
    test_x = []
    test_y = []

    for f in training_file_names[:VALIDATION_GAME_COUNT]:
        data = np.load(f)

        for d in data:
            text_x.append(np.array(d[0]))
            test_y.append(d[1])

    np.save("test_x.npy", test_x)
    np.save("test_y.npy", test_y)

test_x = np.array(test_x)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


if LOAD_PREV_MODEL:
    model = tf.keras.models.load_model(PREV_MODEL_NAME)
else:
    model = Sequential()

    model.add(Conv2D(64, (3,3), padding="same", input_shape=test_x.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

    model.add(Flaten())

    model.add(Dense(64))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))


opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-3)
model.compile(loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=['accuracy'])
