# src/test_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
test_dir = r"C:/Users/Tanmay/OneDrive/Documents/SUMIT NALAVADE/data/test"
IMG_SIZE = (224, 224)

# Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Load Model
model = tf.keras.models.load_model(r"C:\Users\Tanmay\OneDrive\Documents\SUMIT NALAVADE\models\sugarcane_disease_model.keras")

# Evaluate
loss, accuracy = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

