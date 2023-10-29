import numpy as np
import sys
from commpy.channelcoding.ldpc import get_ldpc_code_params, ldpc_bp_decode, triang_ldpc_systematic_encode

design_file = '1440.720.txt'
input_chunk_size = 720
encoded_chunk_size = 1440

# triang_ldpc_systematic_encode
def encode_lim(message_bits):
    
    wimax_ldpc_param = get_ldpc_code_params(design_file) 

    return triang_ldpc_systematic_encode(message_bits, wimax_ldpc_param)


# MSA decode
def decode_lim(coded_bits):
    
    wimax_ldpc_param = get_ldpc_code_params(design_file) 

    coded_bits[coded_bits == 1] = -1
    coded_bits[coded_bits == 0] = 1
    MSA_decoded_bits = ldpc_bp_decode(coded_bits.reshape(-1, order='F').astype(float), wimax_ldpc_param, 'MSA', 10)[0]
    #SPA_decoded_bits = ldpc_bp_decode(coded_bits.reshape(-1, order='F').astype(float), wimax_ldpc_param, 'SPA', 10)[0]

    # Extract systematic part
    MSA_decoded_bits = MSA_decoded_bits[:input_chunk_size].reshape(-1, order='F')
    #SPA_decoded_bits = SPA_decoded_bits[:720].reshape(-1, order='F')

    return MSA_decoded_bits

def encode(message_bits):
    coded_bits = []
    for i in range(0, len(message_bits), input_chunk_size):
        coded_bits=np.concatenate((coded_bits, encode_lim(message_bits[i:min(i+input_chunk_size,len(message_bits))])))
    return coded_bits

def decode(coded_bits):
    coded_bits = np.array(coded_bits)
    if len(coded_bits)%encoded_chunk_size!=0:
        print("LDPC Error: Coded data is not in blocks format")
        sys.exit(1)
    message_bits = []
    for i in range(0, len(coded_bits), encoded_chunk_size):
        message_bits.extend(decode_lim(coded_bits[i:i+encoded_chunk_size]))
    return message_bits


if __name__=="main":
    input = np.random.choice((0,1), int(input_chunk_size))
    input_ = decode(encode(input))
    print(np.equal(input,input_))
