from __future__ import print_function, division
import os
import math

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Concatenate, ReLU, Convolution2D, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.utils import to_categorical

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator

import sys

import numpy as np
import cv2

from keras.preprocessing import image
PATH = os.getcwd()

class DCGAN():
    def __init__(self, batch_size, loss_frequency, sample_frequency, save_frequency):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.data_path = "data-minecraft"
        self.output_path = "output-test"
        self.batch_size = batch_size
        self.loss_frequency = loss_frequency
        self.sample_frequency = sample_frequency
        self.save_frequency = save_frequency
        self.epochs_trained = 0

        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            #horizontal_flip=True,
            dtype="int32"
        )

        self.test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(
            self.data_path + '/train',
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.validation_generator = self.test_datagen.flow_from_directory(
            self.data_path + '/validation',
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        self.num_labels = len(self.train_generator.class_indices)
        self.history_accuracy = []
        self.history_d_loss = []
        self.history_g_loss = []

    def new(self):
        optimizer = Adam(0.0002, 0.5)
        
        # Build the discriminator
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_labels,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        for layer in self.discriminator.layers:
            layer.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], valid)
        self.combined.summary()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def load(self):
        self.discriminator = load_model("CDCGAN-discriminator.h5", compile=False)
        self.generator = load_model("CDCGAN-generator.h5", compile=False)
        self.combined = load_model("CDCGAN-combined.h5", compile=False)

    def build_generator(self):

        noise = Input((self.latent_dim,))
        noise2 = Dense(1024)(noise)
        noise2 = LeakyReLU()(noise2)
		#seems to break it
        #noise2 = Dropout(0.2)(noise2)
        noise2 = Dense(128 * 8 * 8)(noise2)
		#seems to break it
        #noise2 = BatchNormalization()(noise2)
        noise2 = Reshape((8, 8, 128))(noise2)

        label = Input((self.num_labels,))
        label2 = Dense(1024, activation='tanh')(label)
        label2 = Dense(8 * 8 * 128)(label2)
        label2 = BatchNormalization()(label2)
        label2 = Reshape((8, 8, 128))(label2)

        model = Concatenate()([noise2, label2])

        model = UpSampling2D(size=(2, 2))(model)
        model = Conv2D(128, (5, 5), activation='relu', padding='same')(model)

        model = UpSampling2D(size=(2, 2))(model)
        model = Conv2D(64, (5, 5), activation='relu', padding='same')(model)

        model = UpSampling2D(size=(2, 2))(model)
        model = Conv2D(3, (5, 5), activation='tanh', padding='same')(model)

        model = Model([noise, label], model)
        model.summary()
        return model

    def build_discriminator(self):

        img = Input(shape=self.img_shape)
        img2 = Conv2D(64, (5, 5), padding='same', activation='tanh')(img)
        img2 = MaxPooling2D(pool_size=(2, 2))(img2)
        img2 = Conv2D(128, (5, 5), padding='same', activation='tanh')(img2)
        img2 = MaxPooling2D(pool_size=(2, 2))(img2)
        img2 = Flatten()(img2)

        label = Input(shape=((self.num_labels,)))
        label2 = Dense(256, activation='tanh')(label)
        label2 = BatchNormalization()(label2)

        model_input = Concatenate()([img2, label2])
        model = Dense(512, activation='tanh')(model_input)
        model = Dense(1, activation='sigmoid')(model)
        model = Model([img, label], model)
        model.summary()
        return model
    
    def train(self, epochs):

        for epoch in range(epochs):
            batch = self.train_generator.next()
            current_batch_size = len(batch[0])
            training_images = (batch[0] * 2) - 1
            training_labels = batch[1]

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (current_batch_size, self.latent_dim))
            gen_imgs = self.generator.predict([noise, training_labels], current_batch_size)

            valid = np.ones((current_batch_size, 1))
            fake = np.zeros((current_batch_size, 1))

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch([training_images, training_labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, training_labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            valid = np.ones((current_batch_size, 1))
            # Labels to generate images for, random for training
			
            sampled_labels = training_labels
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            self.history_d_loss.append(d_loss[0])
            self.history_accuracy.append(100 * d_loss[1])
            self.history_g_loss.append(g_loss)

            # Plot the progress
            if epoch % self.loss_frequency == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # Save batch and generated images
            if epoch % self.sample_frequency == 0:
                print("Saving images")
                if len(training_images) >= 9:
                    self.save_batch_imgs(epoch, self.train_generator.class_indices, training_images, training_labels)
                
                self.save_sample_imgs(epoch, self.train_generator.class_indices, gen_imgs, sampled_labels)
                print("Images saved")
                print("Saving plot")
                self.save_plot(epoch)
                print("Plot Saved")

            if epoch % self.save_frequency == 0:
                self.save_training(epoch)
    
    def save_plot(self, epoch):
        plt.clf()
        
        accuracy_plot = plt.gca()
        accuracy_plot.plot(self.history_accuracy)
        accuracy_plot.set_title("Model Accuracy")
        accuracy_plot.set_xlabel("Epochs")
        accuracy_plot.set_ylabel("Accuracy")
        accuracy_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(self.output_path + "/%d - accuracy.png" % epoch, dpi=200)
        plt.clf()

        d_loss_plot = plt.gca()
        d_loss_plot.plot(self.history_d_loss)
        d_loss_plot.set_title("Discriminator Loss")
        d_loss_plot.set_xlabel("Epochs")
        d_loss_plot.set_ylabel("Loss")
        d_loss_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(self.output_path + "/%d - d_loss.png" % epoch, dpi=200)
        plt.clf()

        g_loss_plot = plt.gca()
        g_loss_plot.plot(self.history_g_loss)
        g_loss_plot.set_title("Generator Loss")
        g_loss_plot.set_xlabel("Epochs")
        g_loss_plot.set_ylabel("Loss")
        g_loss_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(self.output_path + "/%d - g_loss.png" % epoch, dpi=200)
        plt.clf()

    def save_sample_imgs(self, epoch, label_dict, gen_imgs, labels):

        r, c = 5, 5
        new_noise = np.random.normal(0, 1, size=(r * c, self.latent_dim))
        sampled_labels = np.random.randint(0, self.num_labels, r * c)

        sampled_labels = to_categorical(sampled_labels, self.num_labels)

        gen_imgs = self.generator.predict([new_noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c, gridspec_kw={'wspace':0, 'hspace':0.5})
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                label_dict = self.train_generator.class_indices
                label_categories = sampled_labels[cnt]
                label_id = np.where(label_categories == 1.0)[0]
                label_name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]
                axs[i,j].set_title(label_name, {'fontsize': 10})
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.output_path + "/%d.png" % epoch, dpi=200)
        plt.close()
    
    def save_batch_imgs(self, epoch, label_dict, imgs, labels):
        r = math.floor(math.sqrt(len(imgs)))
        c = r

        #Rescale images 0 - 1
        loaded_imgs = 0.5 * imgs + 0.5
        fig, axs = plt.subplots(r, c, gridspec_kw={'wspace':0, 'hspace':1})
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(loaded_imgs[cnt, :,:,:])
                label_dict = self.train_generator.class_indices
                label_categories = labels[cnt]
                label_id = np.where(label_categories == 1.0)[0]
                label_name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]
                axs[i,j].set_title(label_name, {'fontsize': 10})
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.output_path + "/%d-input.png" % epoch, dpi=200)
        plt.close()

    def save_training(self, epochs):
        print("Saving Models")

        self.generator.save("CDCGAN-generator-test" + str(epochs) + ".h5")

        #self.discriminator.trainable = True
        #for layer in self.discriminator.layers:
        #    layer.trainable = True#

        #self.discriminator.save("CDCGAN-discriminator.h5")

        #self.discriminator.trainable = False
        #for layer in self.discriminator.layers:
            #layer.trainable = False

        #self.combined.get_layer("model_1").trainable = False
        #for layer in self.combined.get_layer("model_1").layers:
            #layer.trainable = False

        #self.combined.save("CDCGAN-combined.h5", include_optimizer=False)
        print("Models saved")
    
def generate_label(label_index, images_to_output, folder_path="output-test/"):
    noise = np.random.normal(0, 1, (images_to_output, dcgan.latent_dim))
    label = np.zeros(shape=(images_to_output, dcgan.num_labels))

    for i in range(0, images_to_output):
        label[i][label_index] = 1.0

    generator = load_model("CDCGAN-generator-working-minecraft-e1600.h5", compile=False)
    
    gen_imgs = generator.predict([noise, label])
    gen_imgs = 0.5 * gen_imgs + 0.5

    folder = folder_path + str(label_index) + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for output_count in range(0, len(gen_imgs)):
        plt.imsave(folder + "gen" + str(output_count) + ".png", gen_imgs[0])

def generate_all_labels(images_to_output, folder_path="output-chris/"):
    for label_index in range(0, dcgan.num_labels):
        generate_label(label_index, images_to_output, folder_path)

def generate_resource_pack(resource_pack_path):

    folder = resource_pack_path + "assets/minecraft/textures/block/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    file = open(resource_pack_path + "pack.mcmeta", "w+")
    file.writelines("{\n")
    file.writelines("\t\"pack\": {\n")
    file.writelines("\t\t\"pack_format\": 4,\n")
    file.writelines("\t\t\"description\": \"" + "test" + "\"\n")
    file.writelines("\t}\n")
    file.writelines("}\n")

    generator = load_model("CDCGAN-generator-working-minecraft-e1600.h5", compile=False)

    for label_index in range(0, dcgan.num_labels):
        noise = np.random.normal(0, 1, (1, dcgan.latent_dim))
        label = np.zeros(shape=(1, dcgan.num_labels))

        label[0][label_index] = 1.0

        gen_imgs = generator.predict([noise, label])
        gen_imgs = 0.5 * gen_imgs + 0.5

        label_dict = dcgan.train_generator.class_indices
        label_categories = label[0]
        label_id = np.where(label_categories == 1.0)[0]
        label_name = list(label_dict.keys())[list(label_dict.values()).index(label_id)]
        
        plt.imsave(folder + label_name + ".png", gen_imgs[0])

if __name__ == '__main__':
    dcgan = DCGAN(batch_size=20, loss_frequency=1, sample_frequency=100, save_frequency=100)
    if len(sys.argv) == 1 or sys.argv[1] == "new":
        dcgan.new()
        dcgan.train(epochs=1000000)
    elif sys.argv[1] == "all":
        generate_all_labels(100)
    elif sys.argv[1] == "minecraft":
        generate_resource_pack("C:/Users/Zephilinox/AppData/Roaming/.minecraft/resourcepacks/test/")
    else:
        desired_label = sys.argv[1]
        label_id = dcgan.train_generator.class_indices[desired_label]
        generate_label(label_id, 100)
