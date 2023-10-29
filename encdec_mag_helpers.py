import cv2
import numpy as np
import tensorflow as tf
import sys
import os

# 0 <-> Negative Value 
# 1 <-> Positive Value
bit_enc = lambda x: -1 if x==0 else 1
bit_dec = lambda x: 0 if x<0 else 1 

# The value that will be changed to embedd the message in a block of size (M,N) is fourier_<color>[block_data_indecies(M,N)]
block_data_indecies = lambda M,N: (M-1, N-1)

def find_ENC_FREQ_MAG(channel_fft, bits):
    best_mag = 90
    min_dist = -1
    for mag in range(99, 20000):
        enc_channel = encode_2_bits(channel_fft, bits, mag)
        dec_bits = decode_2_bits(enc_channel)
        cur_dist = sum(1 for i in range(2) if bits[i] != dec_bits[i])
        if min_dist < 0 or cur_dist < min_dist:
            best_mag = mag
            min_dist = cur_dist
        if min_dist == 0:
            return best_mag
    return best_mag 

def encode_6_bits(image, bits, mag_finder=find_ENC_FREQ_MAG):
    M, N, _ = image.shape
    b, g, r = cv2.split(image)

    fourier_b = np.fft.fft2(b)
    fourier_g = np.fft.fft2(g)
    fourier_r = np.fft.fft2(r)
    
    bits1, bits2, bits3 = np.split(np.array(bits), 3)
    ENC_FREQ_MAG1 = mag_finder(fourier_b, bits1)
    ENC_FREQ_MAG2 = mag_finder(fourier_g, bits2)
    ENC_FREQ_MAG3 = mag_finder(fourier_r, bits3)

    i, j = block_data_indecies(M,N)
    fourier_b[i][j] = complex(bit_enc(bits[0])*ENC_FREQ_MAG1, bit_enc(bits[1])*ENC_FREQ_MAG1)
    fourier_g[i][j]= complex(bit_enc(bits[2])*ENC_FREQ_MAG2, bit_enc(bits[3])*ENC_FREQ_MAG2)
    fourier_r[i][j] = complex(bit_enc(bits[4])*ENC_FREQ_MAG3, bit_enc(bits[5])*ENC_FREQ_MAG3)

    ifourier_b = np.fft.ifft2(fourier_b)
    ifourier_g = np.fft.ifft2(fourier_g)
    ifourier_r = np.fft.ifft2(fourier_r)

    return cv2.merge((
        np.abs(ifourier_b).astype(np.uint8),
        np.abs(ifourier_g).astype(np.uint8),
        np.abs(ifourier_r).astype(np.uint8)
    ))

def decode_6_bits(image):
    M, N, _ = image.shape
    b, g, r = cv2.split(image)

    fourier_b = np.fft.fft2(b)
    fourier_g = np.fft.fft2(g)
    fourier_r = np.fft.fft2(r)

    i, j = block_data_indecies(M,N)
    return [bit_dec(fourier_b[i][j].real), bit_dec(fourier_b[i][j].imag),
            bit_dec(fourier_g[i][j].real), bit_dec(fourier_g[i][j].imag),
            bit_dec(fourier_r[i][j].real), bit_dec(fourier_r[i][j].imag)]



def encode_2_bits(channel_fft, bits, ENC_FREQ_MAG):
    M, N = channel_fft.shape
    i, j = block_data_indecies(M,N)
    channel_fft[i][j] = complex(bit_enc(bits[0])*ENC_FREQ_MAG, bit_enc(bits[1])*ENC_FREQ_MAG)
    
    ifourier = np.fft.ifft2(channel_fft)

    return np.abs(ifourier).astype(np.uint8)

def decode_2_bits(channel):
    M, N = channel.shape

    fourier = np.fft.fft2(channel)
    i, j = block_data_indecies(M,N)
    return [bit_dec(fourier[i][j].real), bit_dec(fourier[i][j].imag)]



def model_file_exists(model_filename):
    return os.path.exists(model_filename)


class MagPredictor:

    def __init__(self, model_filename="trained_model_fft2_to_mag.h5"):
        if model_file_exists(model_filename):
            self.model = tf.keras.models.load_model(model_filename)
            print("Model loaded from " + model_filename) 
        else:
            print("Error: Model file " + model_filename + "couldn't be found")
            sys.exit(1)

    def predict_mag(self, channel_fft, bits):
        res = self.model.predict([np.array([channel_fft.real]), np.array([channel_fft.imag]), np.array([bits])], verbose=0)[0][0]
        #print("predicted: " + str(res))
        return res