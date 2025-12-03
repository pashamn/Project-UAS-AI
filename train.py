import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json, os

# ================================
# 1. Persiapan Directory
# ================================
os.makedirs("models", exist_ok=True)

train_dir = "dataset/train"
val_dir = "dataset/val"

IMG_SIZE = 224
BATCH_SIZE = 32

# ================================
# 2. Data Augmentation (Sangat Penting)
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Simpan mapping class
with open("models/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# ================================
# 3. Transfer Learning MobileNetV2
# ================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Custom classifier
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================
# 4. Callbacks untuk training optimal
# ================================
checkpoint = ModelCheckpoint(
    "models/model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ================================
# 5. Class Weight (atasi dataset tidak seimbang)
# ================================
# Hitung jumlah gambar per class
import numpy as np

counts = np.bincount(train_generator.classes)
max_count = max(counts)
class_weight = {i: max_count / counts[i] for i in range(len(counts))}

print("Class weight digunakan:", class_weight)

# ================================
# 6. Train Model
# ================================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight
)

print("Training selesai. Model terbaik disimpan ke models/model.h5")
