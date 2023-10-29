# Fourier_Image_Stenography
A set of programs that use the 2D-FFT of RGB channels, Error Correcting Codes and Nuaeral Networks to embed and extract files from within an image with maximum performance and accuracy.

TL;DR:
"sten_ldpc_and_fft_model.py" - Uses ldpc with the addition of a Nueral Network based Model that predicts the optimal M for each channel of each block in record time to increase preformance.

Usage:
Embed:
python3 sten_ldpc_and_mag2.py court.jpg -i long_input.txt
Extract:
python3 sten_ldpc_and_mag2.py court_s.bmp -o long_output.txt 


Through this self-study project I developed my own stenography algorithm, improving it with every version.

The basic idea is to divide the image to 2D blocks of uniform size, which are then split to the 3 RGB channels.
Then, For each channel we apply the FFT2, set the complex coefficient corresponding to the highest frequency (To minimize visible changes) to be
complex(+-M,+-M) where M is some positive real number (The sign of each of the real of imaginary components encodes a single bit).
Then, the Inverse Transform is applied to get a valid channel, and saved into the original image.

The output image is saved into the file <original_name_without_ext>_s.bpm, to ensure a lossless compression 
(The accuracy of the extracted file using lossy compression algorithms such as jpg was not checked but might work to some degree)

The decoding process is clear from the description above.

The different "sten" python files correspond to the different ways of choosing M and the varying amounts and types of error correction present, affecting both performance and accuracy:

"sten.py" - Uses arbitrary block size and M, have to be adjusted maunally for each use case

"sten_rs.py" - Uses arbitrary block size and M, have to be adjusted maunally for each use case, with the addition of the Reed-Solomon ECC for bytes which is widely used in noisy channels and Wi-Fi protocols.

"sten_ldpc.py" - Uses arbitrary block size and M, have to be adjusted maunally for each use case, with the addition of the ldpc ECC for bits which is widely used in noisy channels and Wi-Fi protocols.

"sten_mag2.py" - Uses arbitrary block size and chooses the optimal integer M within a certain range for each two bits embedded in each channel of each block using a Brute-Force approach.

"sten_ldpc_and_mag2.py" - Uses ldpc and the mag2 approch. THIS IS THE RECOMMENDED VERSION - I managed to embed the entire "The Sign Of Four" book in 5.5 hours and extract it in 0.5 hours only, WITH NO ERROS.

"sten_ldpc_and_fft_model.py" - Uses ldpc with the addition of a Nueral Network based Model that predicts the optimal M for each channel of each block in record time to increase preformance.

The model, which was designed using keras, was trained by accessing the free random image generation service "Lorem Picsum",
letting it predict the minimal M within a certain range, and comparing its prediction with the actual optimal M calculated in real time.
This approach allows us to train the model witout any local dataset.

The current model is not sufficient for embedding a message properly, but you can further train or change the architecture using the "block_fft2_to_mag_model_train.py" file. 




 
