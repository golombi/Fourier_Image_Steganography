import tensorflow as tf
import numpy as np
from encdec_mag_helpers import find_ENC_FREQ_MAG, model_file_exists
import matplotlib.pyplot as plt
import atexit
import cv2
from image_gen import BlockGen

M, N = 11, 11

model_filename = "trained_model_fft2_to_mag.h5"

def load_or_build_model():
    if model_file_exists(model_filename):
        model = tf.keras.models.load_model(model_filename)
        print("Model loaded from " + model_filename)
    else:
        model = build_model()
        print("Created a new model")
    return model

def build_model():
    input_fft2_real = tf.keras.layers.Input(shape=(M, N))
    input_fft2_imag = tf.keras.layers.Input(shape=(M, N))
    input_binary = tf.keras.layers.Input(shape=(2,))

    # Flatten the real and imaginary parts of the input
    x_real = tf.keras.layers.Flatten()(input_fft2_real)
    x_imag = tf.keras.layers.Flatten()(input_fft2_imag)

    # Concatenate the flattened input with the binary input
    x = tf.keras.layers.concatenate([x_real, x_imag, input_binary])

    # Add more dense hidden layers

    x = tf.keras.layers.Dense(256, activation='relu')(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)

    x = tf.keras.layers.Dense(16, activation='relu')(x)

    # Output layer remains the same
    output = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.models.Model(inputs=[input_fft2_real, input_fft2_imag, input_binary], outputs=output)
    
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

    fft2 = np.array([fourier_r, fourier_g, fourier_b])
    return (fft2.real, fft2.imag, input_binary), output

def train(batch_size, learning_rate=0.0001):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    num_epochs = 100000  # Adjust as needed
    loss_history = []
    
    plt.figure()
    ax = plt.subplot(111)
    
    for epoch in range(num_epochs):
        input_imags = []
        input_reals = []
        input_binaries = []
        outputs = []
        
        for _ in range(batch_size):
            (input_real, input_imag, input_binary), output = gen_pair()
            #input_real_flat = input_real.reshape(-1)
            #input_imag_flat = input_imag.reshape(-1)
            input_reals.extend(input_real)
            input_imags.extend(input_imag)
            input_binaries.extend(input_binary)
            outputs.extend(output)
        
        input_reals = np.array(input_reals)
        input_imags = np.array(input_imags)
        input_binaries = np.array(input_binaries)
        outputs = np.array(outputs)
        
        loss = model.train_on_batch([input_reals, input_imags, input_binaries], outputs)
        
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
    model.save(model_filename)
    print("Model saved to " + model_filename)

if __name__ == '__main__':
    model = load_or_build_model()
    atexit.register(save_model_on_exit, model)
    
    block_gen = BlockGen(M,N)

    train(batch_size=1366)
