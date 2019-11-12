# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:21:52 2019

@author: YQ
"""

import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.d1 = tf.keras.layers.Dense(1024, use_bias=False)
        self.a1 = tf.keras.layers.ReLU()
        self.b1 = tf.keras.layers.BatchNormalization()\
        
        self.d2 = tf.keras.layers.Dense(7*7*128, use_bias=False)
        self.a2 = tf.keras.layers.ReLU()
        self.b2 = tf.keras.layers.BatchNormalization()
        self.r2 = tf.keras.layers.Reshape([7, 7, 128])
        
        self.c3 = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same")
        self.a3 = tf.keras.layers.ReLU()
        self.b3 = tf.keras.layers.BatchNormalization()
        
        self.c4 = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding="same")
        
    def call(self, x, training=True):
        x = self.d1(x)
        x = self.b1(x, training=training)
        x = self.a1(x)
        
        
        x = self.d2(x)
        x = self.b2(x, training=training)
        x = self.a2(x)
        x = self.r2(x)
        
        x = self.c3(x)
        x = self.b3(x, training=training)
        x = self.a3(x)
           
        x = self.c4(x)
        
        x = tf.nn.tanh(x)
        
        return x
    
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.c1 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same")
        self.a1 = tf.keras.layers.LeakyReLU()
        
        self.c2 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same")
        self.a2 = tf.keras.layers.LeakyReLU()
        self.b2 = tf.keras.layers.BatchNormalization()
        self.f2 = tf.keras.layers.Flatten()
        
        self.d3 = tf.keras.layers.Dense(1024)
        self.a3 = tf.keras.layers.LeakyReLU()
        self.b3 = tf.keras.layers.BatchNormalization()
        
        self.D = tf.keras.layers.Dense(1)
        
    def call(self, x, training=True):
        x = self.c1(x)
        x = self.a1(x)
        
        x = self.c2(x)
        x = self.b2(x, training=training)
        x = self.a2(x)
        x = self.f2(x)
        
        x = self.d3(x)
        x = self.b3(x, training=training)
        x = self.a3(x)
        
        mid = x
        
        D = self.D(x)   
        
        return D, mid
    
class QNet(tf.keras.Model):
    def __init__(self):
        super(QNet, self).__init__()
        
        self.Qd = tf.keras.layers.Dense(128)
        self.Qb = tf.keras.layers.BatchNormalization()
        self.Qa = tf.keras.layers.LeakyReLU()
        
        self.Q_cat = tf.keras.layers.Dense(10)
        self.Q_con1_mu = tf.keras.layers.Dense(2)
        self.Q_con1_var = tf.keras.layers.Dense(2)
        self.Q_con2_mu = tf.keras.layers.Dense(2)
        self.Q_con2_var = tf.keras.layers.Dense(2)
        
    def sample(self, mu, var):
        eps = tf.random.normal(shape=mu.shape)
        sigma = tf.sqrt(var)
        z = mu + sigma * eps
        
        return z
    
    def call(self, x, training=True):
        q = self.Qd(x)
        q = self.Qb(x, training=training)
        q = self.Qa(x)
        
        Q_cat = self.Q_cat(q)

        Q_con1_mu = self.Q_con1_mu(q)
        Q_con1_var = tf.exp(self.Q_con1_var(q))
        Q_con2_mu = self.Q_con2_mu(q)
        Q_con2_var = tf.exp(self.Q_con2_var(q))
        
        Q_con1 = self.sample(Q_con1_mu, Q_con1_var)
        Q_con2 = self.sample(Q_con2_mu, Q_con2_var)
        
        return Q_cat, Q_con1, Q_con2

    
if __name__ == "__main__":
    import numpy as np
    
    #tf.debugging.set_log_device_placement(True)
    z = np.random.normal(size=(1, 74)).astype(np.float32)
    z = tf.convert_to_tensor(z)
    
    g = Generator()
    d = Discriminator()
    image = g(z)
    prediction = d(image)

        