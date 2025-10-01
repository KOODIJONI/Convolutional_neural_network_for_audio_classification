import numpy as np
from audio_to_spectrogram import path_to_spectrogram
SPECTROGRAM_SHAPE = (124, 129, 1) 
NUM_LABELS = 8
def resize(inputs, new_height, new_width):
    """Resizing-kerroksen toteutus."""

    _, H, W, _ = inputs.shape
    row_ratio = (H - 1) / (new_height - 1) if new_height > 1 else 0
    col_ratio = (W - 1) / (new_width - 1) if new_width > 1 else 0
    
    resized = np.zeros((1, new_height, new_width, 1), dtype=inputs.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            r = i * row_ratio
            c = j * col_ratio
            r0, c0 = int(np.floor(r)), int(np.floor(c))
            r1, c1 = min(r0 + 1, H - 1), min(c0 + 1, W - 1)
            
            dr, dc = r - r0, c - c0
            
            top_left = inputs[0, r0, c0, 0]
            top_right = inputs[0, r0, c1, 0]
            bottom_left = inputs[0, r1, c0, 0]
            bottom_right = inputs[0, r1, c1, 0]
            
            top = top_left * (1 - dc) + top_right * dc
            bottom = bottom_left * (1 - dc) + bottom_right * dc
            value = top * (1 - dr) + bottom * dr
            
            resized[0, i, j, 0] = value
            
    return resized

#Normalisointi 0 - 1 välille
def normalize(inputs):
    """Normalization-kerroksen toteutus. Laskee keskiarvon ja varianssin itse."""
    
    numpy_mean = inputs.mean(axis=(0,1,2))
    numpy_variance = inputs.var()
    normalized_numpy = (inputs - numpy_mean) / np.sqrt(numpy_variance + 1e-7)

    return normalized_numpy

#Konvoluutio 1
def convolution1(inputs_numpy, filters, biases, strides=(1, 1), padding='VALID', activation='relu'):
    """Toteuttaa konvoluutiolaskennan."""
    
    
    print(inputs_numpy.shape)

    
    input_padded= inputs_numpy
    batch_size, Height, Width, channels = input_padded.shape
    
    kernelH, kernelW, in_channels, out_channels = filters.shape 
    out_H = (Height -kernelH) // strides[0] + 1
    out_W = (Width -kernelW) // strides[1] + 1
   
    output = np.zeros((out_channels,out_H,out_W))

    for f in range(out_channels):
        kernel = filters[:, :, :, f]
        for i in range(out_H):
            for j in range(out_W):
                region = input_padded[0,
                                  i*strides[0]:i*strides[0]+kernelH,
                                  j*strides[1]:j*strides[1]+kernelW,
                                  :]
                value = np.sum(region * kernel)
                if biases is not None:
                    value +=biases[f]
                output[f,i,j] = value
    
    
    manual_output = output.transpose(1, 2, 0) 
    manual_output = np.expand_dims(manual_output, axis=0)
    
    print("Manual output shape:", manual_output.shape)

    
    if activation == 'relu':
        manual_output = np.maximum(manual_output, 0)
    return manual_output



#Maxpooling ns tiivistys juttu vähentää sitä ylioppimista
def maxPooling(inputs, pool_size=(2,2), strides=(2,2)):
        
    N, H, W, C = inputs.shape
    pool_H, pool_W = pool_size
    stride_H, stride_W = strides

    out_H = (H - pool_H) // stride_H + 1
    out_W = (W - pool_W) // stride_W + 1

    output = np.zeros((N, out_H, out_W, C))

    for n in range(N):
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * stride_H
                w_start = j * stride_W
                region = inputs[n, h_start:h_start+pool_H, w_start:w_start+pool_W, :]
                output[n, i, j, :] = np.max(region, axis=(0, 1))
    return output


    
#Litistää inputi vektoriksi.
def flatten(inputs_numpy):

    print("Litistetään...")
    output = inputs_numpy.reshape(inputs_numpy.shape[0], -1)
    return output

def dense1(inputs_numpy, weights, biases, activation='relu'):

    print("Suoritetaan dense 1...")
    manual_output = np.matmul(inputs_numpy, weights) + biases

    if activation == 'relu':
        manual_output = np.maximum(manual_output, 0)
    elif activation == 'softmax':
        exp_outputs = np.exp(manual_output - np.max(manual_output, axis=-1, keepdims=True))
        manual_output = exp_outputs / np.sum(exp_outputs, axis=-1, keepdims=True)
    return manual_output
    
def run_custom_model_with_learned_weights(spectrogram):
    try:
        # Load weights and biases from saved .npy files
        conv1_filters = np.load("weights/layer0_weights.npy")
        conv1_biases  = np.load("weights/layer0_biases.npy")

        conv2_filters = np.load("weights/layer1_weights.npy")
        conv2_biases  = np.load("weights/layer1_biases.npy")

        dense1_weights = np.load("weights/layer4_weights.npy")  # note: layer2 is MaxPooling2D, layer3 is Flatten
        dense1_biases  = np.load("weights/layer4_biases.npy")

        dense2_weights = np.load("weights/layer5_weights.npy")  # last Dense layer
        dense2_biases  = np.load("weights/layer5_biases.npy")

    except (IOError, ValueError) as e:
        print(f"Virhe painojen latauksessa: {e}")
        print("Varmista, että olet ajanut harjoitteluskriptin ja 'my_model_weights.weights.h5'-tiedosto on olemassa.")
        return None

    x = resize(spectrogram, 32, 32)
    x = normalize(x)

    x = convolution1(x, conv1_filters, conv1_biases)
    x = convolution1(x, conv2_filters, conv2_biases)
    x = maxPooling(x)
    x = flatten(x)
    x = dense1(x, dense1_weights, dense1_biases)
    predictions = dense1(x, dense2_weights, dense2_biases)
    
    return predictions
def predict_with_path(path):

  spectrogram = path_to_spectrogram(path)  
  print("error")

  predictions = run_custom_model_with_learned_weights(spectrogram)
  predicted_index = np.argmax(predictions, axis=1)[0]
  predicted_label = label_names[predicted_index]
  print(predicted_label)
  return predicted_label

label_names = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
# if __name__ == "__main__":
#     #lataaa kaikki data
#     try:
#         spectrograms = np.load("spectrograms_data.npy")
#         labels = np.load("labels.npy")
#         num_samples_to_test = 100

#         if len(spectrograms) < num_samples_to_test:
#             print("Datassa on vähemmän kuin 100 näytettä. Testataan kaikilla käytettävissä olevilla näytteillä.")
#             indices = np.arange(len(spectrograms))
#         else:
#             indices = np.random.choice(len(spectrograms), num_samples_to_test, replace=False)
        
#         label_names = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']
#         correct_predictions = 0

#         #ennusta koneoppimismallilla
#         for i, idx in enumerate(indices):
#             test_spectrogram = spectrograms[idx]
#             true_label = labels[idx]
        
#             test_spectrogram = np.expand_dims(test_spectrogram, axis=0) 
        
#             output = run_custom_model_with_learned_weights(test_spectrogram)
        
#             predicted_index = np.argmax(output, axis=1)[0]
        
#             result_str = "✅ Oikein!" if predicted_index == true_label else "❌ Väärin"
#             if predicted_index == true_label:
#                 correct_predictions += 1
        
#             print(f"Näyte {i+1}/{len(indices)}: Todellinen {label_names[true_label]} -> Ennustus {label_names[predicted_index]} ({result_str})")
        
#         accuracy = (correct_predictions / len(indices)) * 100
#         print("\n" + "="*50)
#         print(f"Kokonaisennustustarkkuus: {accuracy:.2f}% ({correct_predictions}/{len(indices)})")
#         print("="*50)

#     except FileNotFoundError:
#         print("spectrograms_data.npy tai labels.npy -tiedostoja ei löydy.")
#         print("Varmista, että olet luonut ne ajamalla datan tallennus- ja harjoitteluskriptin ensin.")