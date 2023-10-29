import cv2
import sys
import numpy as np
import os 
from reedsolo import RSCodec

ENC_FREQ_MAG = 128
BLOCK_WIDTH = 13
BLOCK_HEIGHT = 13
BYTES_FOR_MSG_LEN = 3
#Each chunk of 255 bytes in the RS-encoded message will contain *REDUDENT_BYTES* bytes used for error correction along with up to 255-REDUDENT_BYTES of original message bytes
REDUNDENT_BYTES = 236
print("About " + str(int(REDUNDENT_BYTES/2) / 255 * 100) + " percents of the message (with ECC symbols) can be corrected if altered by noise")

BITS_FOR_MSG_LEN = 8*BYTES_FOR_MSG_LEN
BITS_FOR_MSG_LEN_ECC = 8*(int(np.ceil(BYTES_FOR_MSG_LEN/(255-REDUNDENT_BYTES)))*REDUNDENT_BYTES+BYTES_FOR_MSG_LEN)

BITS_FOR_MSG_ECC_FUNC = lambda msg: 8*(int(np.ceil(len(msg)/(255-REDUNDENT_BYTES)))*REDUNDENT_BYTES+len(msg))
BITS_FOR_MSG_ECC_FUNC_BY_LEN = lambda length: 8*(int(np.ceil(length/(255-REDUNDENT_BYTES)))*REDUNDENT_BYTES+length)

# 0 <-> Negative Value 
# 1 <-> Positive Value
bit_enc = lambda x: -1 if x=='0' else 1
bit_dec = lambda x: '0' if x<0 else '1' 

# The value that will be changed to embedd the message in a block of size (M,N) is fourier_<color>[block_data_indecies(M,N)]
block_data_indecies = lambda M,N: (M-1, N-1)


if len(sys.argv)!=4:
    print("Error: Mismatching argument format: should be; <image_file_name> -i <text_file_to_embedd>(optional) -o <text_file_to_extract_to>(optional)")
    sys.exit(1)

image = cv2.imread(sys.argv[1])
if image is None:
    print("Error: Couldn't read image")
    sys.exit(1)

def bin_string_to_bytearray(str):
    return bytearray(int(str[i:i+8], 2) for i in range(0, len(str), 8))
def len_and_bytes_to_bits(msg):
    # msg is a list of bytes

    # Get string of bits
    len_rs_bytes = bin(len(msg))[2:].zfill(BITS_FOR_MSG_LEN) 
    # Seperate into bytes
    len_rs_bytes = bin_string_to_bytearray(len_rs_bytes)
    # Use RS code such that that BYTES_FOR_MSG_LEN+int(ECC_COEF*BYTES_FOR_MSG_LEN) bytes will be needed for the length field
    len_rs_bytes = RSCodec(REDUNDENT_BYTES).encode(len_rs_bytes) 
    
    # Use RS code such that that len(msg)+int(ECC_COEF*len(msg)) bytes will be needed for the message itself
    msg = RSCodec(REDUNDENT_BYTES).encode(msg)
    bits = []
    for char in len_rs_bytes:
        bits.extend(list(bin(char)[2:].zfill(8)))

    for char in msg:
        bits.extend(list(bin(char)[2:].zfill(8)))
    return bits


def encode_6_bits(image, bits):
    M, N, _ = image.shape
    b, g, r = cv2.split(image)

    fourier_b = np.fft. fft2(b)
    fourier_g = np.fft.fft2(g)
    fourier_r = np.fft.fft2(r)
    
    i, j = block_data_indecies(M,N)
    fourier_b[i][j] = complex(bit_enc(bits[0])*ENC_FREQ_MAG, bit_enc(bits[1])*ENC_FREQ_MAG)
    fourier_g[i][j]= complex(bit_enc(bits[2])*ENC_FREQ_MAG, bit_enc(bits[3])*ENC_FREQ_MAG)
    fourier_r[i][j] = complex(bit_enc(bits[4])*ENC_FREQ_MAG, bit_enc(bits[5])*ENC_FREQ_MAG)

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

    fourier_b = np.fft. fft2(b)
    fourier_g = np.fft.fft2(g)
    fourier_r = np.fft.fft2(r)

    i, j = block_data_indecies(M,N)
    return [bit_dec(fourier_b[i][j].real), bit_dec(fourier_b[i][j].imag),
            bit_dec(fourier_g[i][j].real), bit_dec(fourier_g[i][j].imag),
            bit_dec(fourier_r[i][j].real), bit_dec(fourier_r[i][j].imag)]

def embedd_msg(image, msg):
    M, N, _ = image.shape

    if int(np.log2(8*len(msg)))+1>BITS_FOR_MSG_LEN:
        print("Error: " + str(int(np.log2(8*len(msg)))+1) + " bits are needed to represent the length of the message but " + str(BITS_FOR_MSG_LEN) + " is the limit")
        sys.exit(1)

    bits_capacity = int(M/BLOCK_HEIGHT)*(N/BLOCK_WIDTH)*6
    bits_required = BITS_FOR_MSG_ECC_FUNC(msg)+BITS_FOR_MSG_LEN_ECC
    if bits_required>bits_capacity:
        print("Error: " + str(bits_required) + " bits to embedd with capacity of " + str(bits_capacity))
        sys.exit(1)

    #print("".join(len_and_string_to_bits(msg)))
    bits_itr = iter(len_and_bytes_to_bits(msg))
    for i in range(0, M, BLOCK_HEIGHT):
        for j in range(0, N, BLOCK_WIDTH):
            k=0
            cur_bits=['0','0','0','0','0','0']
            try: 
                while k<6:
                    cur_bits[k] = next(bits_itr)
                    k=k+1
            except StopIteration:
                k=-1
                
            image_block = image[i:min(i+BLOCK_HEIGHT,M), j:min(j+BLOCK_WIDTH, N)]
            #print("".join(cur_bits), end="")
            image[i:min(i+BLOCK_HEIGHT,M), j:min(j+BLOCK_WIDTH, N)] = encode_6_bits(image_block, cur_bits)

            if k<0:
                return
                

def extract_msg(image):
    retreived_bits = 0
    msg_len = -1
    msg_bits = []
    M, N, _ = image.shape
    for i in range(0, M, BLOCK_HEIGHT):
        for j in range(0, N, BLOCK_WIDTH):
            image_block = image[i:min(i+BLOCK_HEIGHT,M), j:min(j+BLOCK_WIDTH, N)]
            msg_bits.extend(decode_6_bits(image_block))
            retreived_bits+=6
            if msg_len<0 and len(msg_bits)>=BITS_FOR_MSG_LEN_ECC:
                msg_len=int.from_bytes(
                            RSCodec(REDUNDENT_BYTES).decode(
                            
                            bin_string_to_bytearray("".join(msg_bits[:BITS_FOR_MSG_LEN_ECC]))
                            
                            )[0]
                        ,byteorder="big")
                msg_bits=msg_bits[BITS_FOR_MSG_LEN_ECC:]
                
            if msg_len>=0 and BITS_FOR_MSG_ECC_FUNC_BY_LEN(msg_len)<=retreived_bits-BITS_FOR_MSG_LEN_ECC:
                msg_bits="".join(msg_bits[:BITS_FOR_MSG_ECC_FUNC_BY_LEN(msg_len)])
                try: 
                    extracted_bytes = RSCodec(REDUNDENT_BYTES).decode(bin_string_to_bytearray(msg_bits))[0]
                except ValueError:
                    print("Can't extract message => some byte is not in range [0,255]")
                    sys.exit(1)
                return extracted_bytes

if sys.argv[2]=="-o":
    try:
        with open(sys.argv[3], "wb") as output_file:
            output_file.write(extract_msg(image))
    except FileNotFoundError:
        print("Error: file " + sys.argv[2] + " not found")
        sys.exit(1)
    except Exception as e:
        print(f"Exctraction Error: {e}")

elif sys.argv[2]=="-i":
    try:
        with open(sys.argv[3], "rb") as input_file:
            input_bytes = input_file.read()
    except FileNotFoundError:
        print("Error: file " + sys.argv[2] + " not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
    embedd_msg(image, input_bytes)
    # Save the edited image as a BMP file
    original_name, file_extension = os.path.splitext(sys.argv[1])
    output_file_name = original_name + "_s.bmp"
    cv2.imwrite(output_file_name, image)

