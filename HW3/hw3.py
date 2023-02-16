import numpy as np
import io
import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks


# Check tensorflow version and GPU available
print(tf.__version__)
print(tf.test.is_gpu_available())

# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='predict',help='train/predict')
ap.add_argument('--weights', default='weights/model.h5', help='model.h5 path')
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--MemoryCell', type=str, default='LSTM')
ap.add_argument('--ckpt_dir', type=str, default='tmp/checkpoints')
opt = ap.parse_args()
EPOCHS = opt.epochs
weights_dir = opt.weights
mode = opt.mode
MemoryCell = opt.MemoryCell
ckpt_dir = opt.ckpt_dir


### ================================================ Read Data ================================================== ###
print("== Read Data ================================================================")
data_URL = 'shakespeare_train.txt'
val_URL = 'shakespeare_valid.txt'
with io.open(data_URL, 'r', encoding = 'utf8') as f:
    text = f.read()
with io.open(val_URL, 'r', encoding = 'utf8') as f:
    val_text = f.read()

# Characters'collection
vocab = sorted(set(text))

# Construct character dictionary
vocab_to_int = {c:i for i, c in enumerate(vocab)}
# int_to_vocab = dict(enumerate(vocab))
int_to_vocab = np.array(vocab)

# Encodedata, shape=[number of characters]
train_data = np.array([vocab_to_int[c] for c in text], dtype = np.int32)
valid_data = np.array([vocab_to_int[c] for c in val_text], dtype = np.int32)
print("Len of train data: {}".format(len(train_data)))
print("Len of valid data: {}".format(len(valid_data)))
# print(train_data[:10])
# print(valid_data[:100])
# for i in valid_data[:100]:
#     print(int_to_vocab[i], end='')
# print()

# Analyize the dataset
print('Length of train text: {} characters'.format(len(text)))
print('Length of valid text: {} characters'.format(len(val_text)))

print('{} unique characters'.format(len(vocab)))
print('vocab:', vocab)

# print('{')
# for char, _ in zip(vocab_to_int, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), vocab_to_int[char]))
# print('  ...\n}')


### =========================================== Create training sequences ============================================= ###

print("== Create training sequences ================================================")
# The maximum length sentence we want for a single input in characters.
sequence_length = 50
steps_per_epoch = len(train_data) // (sequence_length + 1)
validation_step = len(valid_data) // (sequence_length + 1)

print('steps_per_epoch:', steps_per_epoch)
print('validation_step:', validation_step)

# Create training dataset and validation dataset.
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)

# for char in train_dataset.take(5):
#     print(int_to_vocab[char.numpy()])

# Generate batched sequences out of the train_dataset. (Sequences size is the same as examples_per_epoch)
train_sequences = train_dataset.batch(sequence_length + 1, drop_remainder=True)
valid_sequences = valid_dataset.batch(sequence_length + 1, drop_remainder=True)

# Sequences examples.
print()
for item in train_sequences.take(5):
    print(repr(''.join(int_to_vocab[item.numpy()])))
print()

# Create an sequence of "input data" and "target output"
# Ex: The input sequence would be "Hell", and the target sequence "ello".
def get_input_output_split(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

train_dataset = train_sequences.map(get_input_output_split)
valid_dataset = valid_sequences.map(get_input_output_split)
# Dataset size is the same as examples_per_epoch.
# But each element of a sequence is now has length of `sequence_length`
# and not `sequence_length + 1`.
# print('dataset size: {}'.format(len(list(train_dataset.as_numpy_iterator()))))

# Check "input data" and "target output"
for input_example, target_example in train_dataset.take(1):
    print('Input sequence size:', repr(len(input_example.numpy())))
    print('Target sequence size:', repr(len(target_example.numpy())))
    print()
    print('Input:', repr(''.join(int_to_vocab[input_example.numpy()])))
    print('Target:', repr(''.join(int_to_vocab[target_example.numpy()])))

# for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
#     print('Step {:2d}'.format(i))
#     print('  input: {} ({:s})'.format(input_idx, repr(int_to_vocab[input_idx])))
#     print('  expected output: {} ({:s})'.format(target_idx, repr(int_to_vocab[target_idx])))


### ==================================== Split training sequences into batches ====================================== ###
## We used tf.data to split the text into manageable sequences. 
## But before feeding this data into the model, we need to shuffle the data and pack it into batches.

print("== Split training sequences into batches ========================================")

BATCH_SIZE = 64

# Buffer size to shuffle the dataset (TF data is designed to work
# with possibly infinite sequences, so it doesn't attempt to shuffle
# the entire sequence in memory. Instead, it maintains a buffer in
# which it shuffles elements).
BUFFER_SIZE = 10000
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
valid_dataset = valid_dataset.batch(BATCH_SIZE, drop_remainder=True)
print(train_dataset)
print('Batched dataset size: {}'.format(len(list(train_dataset.as_numpy_iterator()))))
for input_text, target_text in train_dataset.take(1):
    print('1st batch: input_text:', input_text)
    print()
    print('1st batch: target_text:', target_text)



if mode == 'train':

    ### ==================================== Build RNN model ====================================== ###
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import Sequential      # 啟動NN
    from tensorflow.keras.layers import Embedding       # Embedding layer
    from tensorflow.keras.layers import LSTM, SimpleRNN
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.initializers import GlorotNormal
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.losses import sparse_categorical_crossentropy
    from tensorflow.keras.callbacks import ModelCheckpoint

    #  if softmax layer is not being added at the last layer,
    #  then we need to have the from_logits=True to indicate the probabilities are not normalized
    def loss(labels, logits):
        return sparse_categorical_crossentropy(
        y_true=labels,
        y_pred=logits,
        from_logits=True
        )

    # Length of the vocabulary in chars. (how many different characters)
    vocab_size = len(vocab) 

    # Build model
    model = Sequential()

    if MemoryCell == "LSTM":
        model.add(Embedding(input_dim=vocab_size, output_dim=256, batch_input_shape=[BATCH_SIZE, None]))
        model.add(LSTM(units=1024, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        # model.add(LSTM(units=256, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        model.add(Dense(units=vocab_size))

    elif MemoryCell == "RNN":
        model.add(Embedding(input_dim=vocab_size, output_dim=256, batch_input_shape=[BATCH_SIZE, None]))
        model.add(SimpleRNN(units=1024, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        # model.add(SimpleRNN(units=256, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        model.add(Dense(units=vocab_size))
    
    else:
        exit(1) # error and stop the program

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss)

    model.summary()
    plot_model(model=model, to_file='run/epochs/model.png', show_shapes=True, show_layer_names=True)

    for input_example_batch, target_example_batch in train_dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


    # Checkpoint

    checkpoint = ModelCheckpoint("run/epochs/best_weights.h5", 
                                monitor='val_loss', verbose=1, save_best_only=True, 
                                mode='auto', save_freq='epoch', save_weights_only=True)

    # Directory where the checkpoints will be saved.
    checkpoint_dir = 'tmp/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}.h5')

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    # Training
    model_info = model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=valid_dataset,
        callbacks=[checkpoint_callback, checkpoint]
    )

    # 儲存權重
    model.save_weights('run/epochs/last_weights.h5')
    model.save('run/epochs/model_keras.h5')
    print('\n====================================================')
    print('Weights are saved to run/train-face/epochs/last_weights.h5')
    print('Best weights are saved to run/train-face/epochs/best_weights.h5')
    print('====================================================\n')

    # 繪製訓練 & 驗證的損失值
    plt.figure()
    plt.grid()
    plt.plot(model_info.history['loss'])
    plt.plot(model_info.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('run/epochs/History_loss.png')
    plt.show()

elif mode == 'test':
    from tensorflow.keras.models import Sequential      # 啟動NN
    from tensorflow.keras.layers import Embedding       # Embedding layer
    from tensorflow.keras.layers import LSTM, SimpleRNN
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.initializers import GlorotNormal

    
    checkpoint_dir = ckpt_dir

    simplified_batch_size = 1
    vocab_size = len(vocab) 

    # Build model
    model = Sequential()

    if MemoryCell == "LSTM":
        model.add(Embedding(input_dim=vocab_size, output_dim=256, batch_input_shape=[simplified_batch_size, None]))
        model.add(LSTM(units=1024, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        # model.add(LSTM(units=256, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        model.add(Dense(units=vocab_size))

    elif MemoryCell == "RNN":
        model.add(Embedding(input_dim=vocab_size, output_dim=256, batch_input_shape=[simplified_batch_size, None]))
        model.add(SimpleRNN(units=1024, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        # model.add(SimpleRNN(units=256, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        model.add(Dense(units=vocab_size))
    
    else:
        exit(1) # error and stop the program

    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    # model.load_weights('run/epochs_40/last_weights.h5')
    model.build(tf.TensorShape([simplified_batch_size, None]))
    model.summary()

    def predict_text(model, start_string, num_generate = 400):
        '''
        num_generate
        - number of characters to generate.
        '''
        # Evaluation step (generating text using the learned model)

        # Converting our start string to numbers (vectorizing).
        input_indices = [vocab_to_int[s] for s in start_string]
        input_indices = tf.expand_dims(input_indices, 0)

        # Empty string to store our results.
        text_generated = []

        # Here batch size == 1.
        model.reset_states()
        for char_index in range(num_generate):
            predictions = model(input_indices)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            # print("shape of predictions: {}".format(predictions.shape))

            # Using a categorical distribution to predict the character returned by the model.
            predicted_id = np.argmax(predictions[-1].numpy())
            # print(predicted_id)

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state.
            input_indices = tf.expand_dims([predicted_id], 0)

            text_generated.append(int_to_vocab[predicted_id])

        # print("Last prediction:\n",predictions)
        # print("predicted_id: {}, value: {:.4f}".format(predicted_id, predictions[0, predicted_id]))
        return (start_string + ''.join(text_generated))
    
    # Predict some text from different epoch
    for i in np.arange(0, EPOCHS, EPOCHS / 5):
        print("\n================================================================")
        print("For epoch {}\n".format(int(i+1)))
        ckpt = checkpoint_dir + '/ckpt_' + str(int(i+1)) + '.h5'
        model.load_weights(ckpt)
        print(predict_text(model, start_string=u"First Citizen:\nBefore we proceed any further, hear me speak."))


elif mode == 'generate':

    from tensorflow.keras.models import Sequential      # 啟動NN
    from tensorflow.keras.layers import Embedding       # Embedding layer
    from tensorflow.keras.layers import LSTM, SimpleRNN
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.initializers import GlorotNormal


    checkpoint_dir = ckpt_dir
    tf.train.latest_checkpoint(checkpoint_dir)

    simplified_batch_size = 1
    vocab_size = len(vocab) 

    # Build model
    model = Sequential()

    if MemoryCell == "LSTM":
        model.add(Embedding(input_dim=vocab_size, output_dim=256, batch_input_shape=[simplified_batch_size, None]))
        model.add(LSTM(units=1024, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        # model.add(LSTM(units=256, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        model.add(Dense(units=vocab_size))

    elif MemoryCell == "RNN":
        model.add(Embedding(input_dim=vocab_size, output_dim=256, batch_input_shape=[simplified_batch_size, None]))
        model.add(SimpleRNN(units=1024, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        # model.add(SimpleRNN(units=256, activation='tanh', return_sequences=True, stateful=True, recurrent_initializer=GlorotNormal()))
        model.add(Dense(units=vocab_size))
    
    else:
        exit(1) # error and stop the program

    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.load_weights('run/epochs_30_LSTM/best_weights.h5')
    model.build(tf.TensorShape([simplified_batch_size, None]))

    model.summary()

    
    def generate_text(model, start_string, num_generate = 1000, temperature=1.0):
        '''
        num_generate
        - number of characters to generate.
        
        temperature
        - Low temperatures results in more predictable text.
        - Higher temperatures results in more surprising text.
        - Experiment to find the best setting.
        '''
        # Evaluation step (generating text using the learned model)

        # Converting our start string to numbers (vectorizing).
        input_indices = [vocab_to_int[s] for s in start_string]
        input_indices = tf.expand_dims(input_indices, 0)

        # Empty string to store our results.
        text_generated = []

        # Here batch size == 1.
        model.reset_states()
        for char_index in range(num_generate):
            predictions = model(input_indices)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # Using a categorical distribution to predict the character returned by the model.
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state.
            input_indices = tf.expand_dims([predicted_id], 0)

            text_generated.append(int_to_vocab[predicted_id])

        print("Last prediction:\n",predictions)
        print("predicted_id: {}, value: {:.4f}".format(predicted_id, predictions[0, predicted_id]))
        return (start_string + ''.join(text_generated))
    
    # Generate the text with default temperature (1.0).
    print(generate_text(model, start_string=u"ROMEO:"))

    model_name = 'weights/text_generation_shakespeare_rnn.h5'
    model.save(model_name, save_format='h5')