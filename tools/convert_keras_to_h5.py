import tensorflow as tf
from tensorflow import keras
from keras import layers



# Or, if it's an older format:
loaded_model = keras.models.load_model('model_export.keras')

# Then save it to a new .h5 file
loaded_model.save('model_export.h5')