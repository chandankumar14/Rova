import numpy as np
import tensorflow as tf
MODEL = tf.keras.models.load_model("./Trained_Model/Tomato")
CLASS_NAMES = [
'Tomato_Bacterial_spot',
 'Tomato_Early_blight_',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites',
 'Tomato_Target_Spot',
 'Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy'
 ]

def TomatoModel(image):
    img_batch = np.expand_dims(image, 0)
    predictions =  MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return {
        'class': predicted_class,
        'accuracy': confidence 
    }

