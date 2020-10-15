# -----------------------------------------------------------------------------------------------------------
# ============================================ REQUIRED IMPORTS =============================================
# -----------------------------------------------------------------------------------------------------------

# ====== Regular Imports ======
import os
import numpy as np
import scipy as sp
import pandas as pd
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# ======= Visualization ========
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# ======= Images ========
import VisualizerClass
import cv2
from glob import glob
from PIL import Image
import albumentations as alb

# ========= SKLearn =========
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# ======== Transfer Learnin ======
from efficientnet.tfkeras import *
from tensorflow.keras.applications import ResNet50, Xception

# ======== TensorFlow ========
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.python.keras.layers.advanced_activations import *
from tensorflow.keras.initializers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.constraints import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

print("All libraries successfully imported.")



# -----------------------------------------------------------------------------------------------------------
# ========================================= ORGANIZING TABULAR DATA =========================================
# -----------------------------------------------------------------------------------------------------------

base_path = "C:/Users/harsh/Desktop/Python/Projects/MRI Segmentation"
base_path2 = "D:/Datasets/Brain Cancer Detection"
data = pd.read_csv(f"{base_path}/kaggle_3m/data.csv")


# -----------------------------------------------------------------------------------------------------------
# ========================================== INSPECTING IMAGE DATA ==========================================
# -----------------------------------------------------------------------------------------------------------

# sample = f"{base_path}/kaggle_3m/TCGA_CS_4942_19970222/TCGA_CS_4942_19970222_1.tif"
# sample_mask = f"{base_path}/kaggle_3m/TCGA_CS_4942_19970222/TCGA_CS_4942_19970222_1_mask.tif"

# sample_img = cv2.imread(sample)
# plt.imshow(sample_img)


# -----------------------------------------------------------------------------------------------------------
# ========================================== ORGANIZING IMAGE DATA ==========================================
# -----------------------------------------------------------------------------------------------------------

def extract_image_paths(base_path):
    train_files = []
    mask_files = glob(f'{base_path}/kaggle_3m/*/*_mask*')
    mask_files = [mask.replace("\\", "/") for mask in mask_files]

    for file in mask_files:
        train_files.append(file.replace("_mask", ""))

    return train_files, mask_files

def make_dataset(base_path, test_size, val_size):
    train_files, mask_files = extract_image_paths(base_path)

    df = pd.DataFrame({'images': train_files, 'masks': mask_files})
    train_set, test_set = train_test_split(df, test_size=test_size)
    train_set, val_set = train_test_split(train_set, test_size=val_size)

    return df, train_set, test_set, val_set


# -----------------------------------------------------------------------------------------------------------
# =========================================== IMAGE PREPROCESSING ===========================================
# -----------------------------------------------------------------------------------------------------------

def get_image_resolution(image):
    img = cv2.imread(image)
    height, width, clr_channel = img.shape
    resolution = [height, width, clr_channel]

    return resolution

def scan_image_abnormalities(base_path, base_img_resolution, base_msk_resolution):
    """[Scans directory for abnormalities in images]
    Args:
        base_path ([str]): [parent directory - the directory above the dataset dir]
        base_img_resolution([list]): [list in format - [height, width, channel] ]
        base_msk_resolution([list]): [list in format - [height, width, channel] ]

    Returns:
        [abnormal_image_properties]: [list of abnormal images and their heights, widths, and channels]
        [abnormal_mask_properties]: [list of abnormal masks and their heights, widths, and channels]
    """

    # Defining lists for appending paths of abnormal images, and their heights, widths, and channels
    ab_imgs, ab_im_h, ab_im_w, ab_im_c = [], [], [], []
    ab_masks, ab_msk_h, ab_msk_w, ab_msk_c = [], [], [], []

    train_files, mask_files = extract_image_paths(base_path)
    
    for img, mask in zip(train_files, mask_files):
        img_resolution = get_image_resolution(img)
        msk_resolution = get_image_resolution(mask)     

        if img_resolution != base_img_resolution:
            ab_imgs.append(img)
            if img_resolution[0] != 500: ab_im_h.append(img_resolution[0])
            elif img_resolution[1] != 500:ab_im_w.append(img_resolution[1])
            elif img_resolution[2] != 3: ab_im_c.append(img_resolution[1])

        if msk_resolution != base_msk_resolution:
            ab_masks.append(mask)
            if msk_resolution[0] != 500: ab_msk_h.append(msk_resolution[0])
            elif msk_resolution[1] != 500:ab_msk_w.append(msk_resolution[1])
            elif msk_resolution[2] != 3: ab_msk_c.append(msk_resolution[1])
        
    abnormal_image_properties = [ab_imgs, ab_im_h, ab_im_w, ab_im_c]
    abnormal_mask_properties = [ab_masks, ab_msk_h, ab_msk_w, ab_msk_c]

    return abnormal_image_properties, abnormal_mask_properties 


    # -----------------------------------------------------------------------------------------------------------
# =========================================== IMAGE-DATA GENERATOR ==========================================
# -----------------------------------------------------------------------------------------------------------

def normalize_and_highlight(highlight, img, mask):
    # Normalizing
    img = img / 255
    mask = mask / 255

    # Darkening/Lightening highlights 
    if highlight:
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    
    return (img, mask)


# Adopted From: https://github.com/zhixuhao/unet/blob/master/data.py
def imagedatagenerator(df, batch_size, augmentations, target_size, highlight,
                    image_color_mode="rgb", mask_color_mode="grayscale",
                    image_save_prefix="image", mask_save_prefix="mask",
                    save_to_dir=None, seed=777):
    
    image_datagen = ImageDataGenerator(**augmentations)
    mask_datagen = ImageDataGenerator(**augmentations)
    
    image_generator = image_datagen.flow_from_dataframe(
        df, x_col = "images",
        shuffle = False,
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        df, x_col = "masks",
        shuffle = False,
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    
    for (img, mask) in zip(image_generator, mask_generator):
        img, mask = normalize_and_highlight(highlight, img, mask)
        yield (img,mask)


def call_and_define_generators(augment, train_set, val_set, 
                               batch_size, augmentations, target_size, highlight):

    if augment:
        train_generator = imagedatagenerator(train_set, batch_size, 
                                            augmentations, target_size, highlight)
        test_generator = imagedatagenerator(val_set, batch_size, 
                                            dict(), target_size, highlight)
    
    else: 
        train_generator = imagedatagenerator(train_set, batch_size, 
                                            dict(), target_size, highlight)
        test_generator = imagedatagenerator(val_set, batch_size, 
                                            dict(), target_size, highlight)
    
    return train_generator, test_generator

# -----------------------------------------------------------------------------------------------------------
# =========================================== MODEL CALLBACKS ===============================================
# -----------------------------------------------------------------------------------------------------------

def reducelronplateau():
    reducelronplateau = ReduceLROnPlateau(
        monitor='loss', factor=0.05, 
        patience=5, verbose=1, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=0.001)

    return reducelronplateau

def modelcheckpoint(checkpoint_filepath):
    modelcheckpoint = ModelCheckpoint(
        filepath=checkpoint_filepath, 
        save_weights_only=True, save_best_only=True,
        monitor='val_acc', mode='max')
    
    return modelcheckpoint

def earlystopping():    
    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, 
        verbose=0, mode='auto', baseline=None, 
        restore_best_weights=False)
    
    return earlystopping

def get_model_callbacks(model_checkpoint_filepath):
    callbacks = [
        reducelronplateau(),
        modelcheckpoint(model_checkpoint_filepath),
        earlystopping()
    ]

    return callbacks

# -----------------------------------------------------------------------------------------------------------
# ============================================= MODEL OPTIMIZERS ============================================
# -----------------------------------------------------------------------------------------------------------

def get_mixed_precision_opt(optimizer):
    return tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)


# -----------------------------------------------------------------------------------------------------------
# ========================================= IMAGE SEGMENTATION MODEL ========================================
# -----------------------------------------------------------------------------------------------------------

def get_unet(input_shape):
    inputs = Input(input_shape)

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    UNet = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return UNet

# =================================== CUSTOM LOSSES AND METRICS ==========================
smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return -iou(y_true, y_pred)

def dice_loss(y_true, y_pred):
    numerator = 2 * (tf.math.abs(tf.sets.intersection(y_true, y_pred).astype("float16")))
    denominator = tf.math.abs(y_true) + tf.math.abs(y_pred)
    diceLoss = numerator/denominator
    return diceLoss

# -----------------------------------------------------------------------------------------------------------
# ================================= PATHS, PARAMETERS, AND MODEL TUNING =====================================
# -----------------------------------------------------------------------------------------------------------
# RUNTIME NAME ***
runtime_name = "trial1"  # MUST change this every time the code is run

# Model Hyperparameters
test_size, val_size = 0.1, 0.2
EPOCHS = 5
batch_size = 32
learning_rate = 0.0001
input_shape = (256, 256, 3)
target_size = (256, 256)

# Data-Generator Parameters
highlight = True
augment = True
model_checkpoint_filepath = f"{base_path}/Model_Checkpoints/{runtime_name}.hdf5"

# Data-Generator Augmentations
augmentations = dict(
    rotation_range=0.25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=(0.45, 0.80),
    fill_mode='nearest',
    shear_range=0.1,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
)

# Adam Optimizer arguments
beta_1, beta_2 = 0.9, 0.999
epsilon, amsgrad = 1e-07, False

# Loss and Metrics
loss = dice_coef_loss
metrics = ['binary_accuracy', dice_coef]

# ============================
# dataframes
print(f"\n {'-' * 50} \n COLLECTING DATAFRAMES... \n {'-' * 50} ")
df, train_set, test_set, val_set = make_dataset(base_path, test_size, val_size)
print(f"\n {'-' * 50} \n SUCCESSFULLY COLLECTED DATAFRAMES \n {'-' * 50} ")

# optimizer 
optimizer = Adam(
    learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, 
    epsilon=epsilon, amsgrad=amsgrad, name='Adam'
)

mixed_precision_optimizer = get_mixed_precision_opt(optimizer)

# compiling model
print(f"\n {'-' * 50} DEFINING U-NET... \n {'-' * 50}")
UNet = get_unet(input_shape)
print(f"\n {'-' * 50} U-NET SUCCESSFULLY CREATED \n {'-' * 50}")

print(f"\n {'-' * 50} COMPILING U-NET... \n {'-' * 50}")
UNet.compile(optimizer=mixed_precision_optimizer, loss=loss, metrics=metrics)
print(f"\n {'-' * 50} U-NET SUCCESSFULLY COMPILED \n {'-' * 50}")

# Data Generator
print(f"\n {'-' * 50} PREPARING THE IMAGE-DATA-GENERATOR")
train_generator, test_generator = call_and_define_generators(
    augment, train_set, val_set, batch_size, augmentations, target_size, highlight
)

# Training model
model_checkpoint_filepath = f"{base_path}/Model_Checkpoints/{runtime_name}.hdf5"
callbacks = get_model_callbacks(model_checkpoint_filepath)

print("\n {'-' * 50} BEGINNING MODEL TRAINING... \n {'-' * 50}")
history = UNet.fit(
    train_generator,
    epochs = EPOCHS,
    # callbacks = callbacks,
    validation_data = test_generator,
    validation_steps = 10    
)

print(f"\n {'-' * 50} MODEL SUCCESSFULLY TRAINED \n {'-' * 50}")


# Saving Model
model_path = f"{base_path}/{models_dir}/{runtime_name}.hdf5"
model.save(model_path)

print(f"\n {'-' * 50} MODEL SAVED TO {model_path} \n {'-' * 50}")


# Visualizing model progress through the training loop
plot_path = f"{model_path.split(sep='.')[0]}.jpg"

history = history.history
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=['Loss', 'Accuracy'])

fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history['loss'],
                        mode='lines+markers', name='Loss'), 
                        row=1, col=1)

fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history['accuracy'],
                        mode='lines+markers', name='Accuracy'), 
                        row=1, col=2)

fig.update_xaxes(title_text='Epochs', row=1, col=1)
fig.update_xaxes(title_text='Epochs', row=1, col=2)
    
fig.update_layout(title=f"Visualizing Model's Progress")

fig.write_image(plot_path) 

print(f"\n {'-' * 50} PLOT OF MODEL PROGRESS SAVED AT {plot_path}")

# Loading model for inference
print(f"\n {'-' * 50} LOADING MODEL FOR EVALUATION... \n {'-' * 50}")
trained_model = load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

eval_data_generator = imagedatagenerator(test_set, batch_size, 
                                        dict(), target_size=target_size)

print(f"\n {'-' * 50} BEGINNING MODEL EVALUATION \n {'-' * 50}")
scores = saved_model.evaluate(eval_data_generator, steps = len(test_set)/batch_size)
loss, iou, dice_coef = scores[0], scores[1], scores[3]

print(f"{'-' * 50} MODEL SUCCESSFULLY EVALUATED ")
print(f"Model loss: {loss}")
print(f"Model IOU: {iou}")
print(f"Model Dice Coefficient: {dice_coef} \n{'-' * 50}")
