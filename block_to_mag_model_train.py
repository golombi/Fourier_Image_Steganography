import tensorflow as tf
import numpy as np
from encdec_mag_helpers import find_ENC_FREQ_MAG
import matplotlib.pyplot as plt
import atexit
import os
import cv2
import sys
from image_gen import BlockGen

M, N = 11, 11

def model_file_exists():
    return os.path.exists("trained_model_on_exit.h5")

def load_or_build_model():
    if model_file_exists():
        model = tf.keras.models.load_model("trained_model_on_exit.h5")
        print("Model loaded from trained_model_on_exit.h5")
    else:
        model = build_model()
        print("Created a new model")
    return model

def build_model():
    input_channel = tf.keras.layers.Input(shape=(M, N))

    input_binary = tf.keras.layers.Input(shape=(2,))

    x_channel = tf.keras.layers.Flatten()(input_channel)

    # Concatenate the flattened input with the binary input
    x = tf.keras.layers.concatenate([x_channel, input_binary])

    # Add more dense hidden layers

    x = tf.keras.layers.Dense(256, activation='relu')(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # Output layer remains the same
    output = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.models.Model(inputs=[input_channel, input_binary], outputs=output)
    
    return model



def gen_pair():
    input_image = block_gen.gen_block()
    input_binary = np.split(np.random.randint(2, size=(6,)), 3)
    
    b,g,r = cv2.split(input_image)
    fourier_b = np.fft.fft2(b)
    fourier_g = np.fft.fft2(g)
    fourier_r = np.fft.fft2(r)

    
    output = np.array([find_ENC_FREQ_MAG(fourier_r, input_binary[0]),
                       find_ENC_FREQ_MAG(fourier_g, input_binary[1]),
                       find_ENC_FREQ_MAG(fourier_b, input_binary[2])])
    
    return (np.array([r,g,b]), input_binary), output

def train(batch_size, learning_rate=0.001):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    num_epochs = 100000  # Adjust as needed
    loss_history = []
    
    plt.figure()
    ax = plt.subplot(111)
    
    for epoch in range(num_epochs):
        input_channels = []
        input_binaries = []
        outputs = []
        
        for _ in range(batch_size):
            (input_channel, input_binary), output = gen_pair()


            input_channels.extend(input_channel)
            input_binaries.extend(input_binary)
            outputs.extend(output)
        
        input_channels = np.array(input_channels)
        input_binaries = np.array(input_binaries)
        outputs = np.array(outputs)
        
        loss = model.train_on_batch([input_channels, input_binaries], outputs)
        
        loss_history.append(loss)
        
        if epoch % 1 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss}')
            ax.clear()
            ax.plot(range(len(loss_history)), loss_history)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.pause(0.1)

def save_model_on_exit(model):
    model.save("trained_model_on_exit.h5")
    print("Model saved to trained_model_on_exit.h5")

if __name__ == '__main__':
    model = load_or_build_model()
    atexit.register(save_model_on_exit, model)
    
    block_gen = BlockGen(M,N)

    train(batch_size=341)
