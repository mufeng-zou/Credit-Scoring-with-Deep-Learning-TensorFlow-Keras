# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:31:10 2017

@author: mxz20
"""

import numpy as np
import pandas as pd
from keras.layers import Input, Dense, LSTM, Masking, concatenate, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math

def read_sample():
    global pd_train, context_train, gbf_train, cf_train, rf_train, rq_train, dj_train, wr_train, cq_train
    global pd_test, context_test, gbf_test, cf_test, cf_test, rf_test, rq_test, dj_test, wr_test,cq_test
    
    pd_train = np.load('../data/pd_train.npy')
    pd_test = np.load('../data/pd_test.npy')
    
    context_train = np.load('../data/context_train.npy')
    context_test = np.load('../data/context_test.npy')
    
    cf_train = np.load('../data/cf_train.npy')
    cf_test = np.load('../data/cf_test.npy')
    
    rf_train = np.load('../data/rf_train.npy')
    rf_test = np.load('../data/rf_test.npy')
    
    rq_train = np.load('../data/rq_train.npy').reshape(39977, 20, 50)
    rq_test = np.load('../data/rq_test.npy').reshape(10023, 20, 50)
    
    dj_train = np.load('../data/dj_train.npy')
    dj_test = np.load('../data/dj_test.npy')
    
    gbf_train = np.load('../data/gbf_train.npy')
    gbf_test = np.load('../data/gbf_test.npy')
    
    wr_train = np.load('../data/wr_train.npy').reshape(39977,4)
    wr_test = np.load('../data/wr_test.npy').reshape(10023,4)
    
    cq_train = np.load('../data/cq_train.npy').reshape(39977, 10, 48)
    cq_test = np.load('../data/cq_test.npy').reshape(10023, 10, 48)
    
if __name__ == "__main__":
    
    
    read_sample()
    
    np.random.seed(7)
    
    #model structure
    pd_input = Input(shape=(5,), dtype='float32', name='pd_input')
    pd_norm = BatchNormalization()(pd_input)
    pd_relu = Dense(5, activation='relu')(pd_norm) #change
    
    context_input = Input(shape=(33,), dtype='float32', name='context_input')
    context_norm = BatchNormalization()(context_input)
    context_relu = Dense(20, activation='relu')(context_norm) #change
    
    cf_input = Input(shape=(4,26), dtype='float32', name='cf_input')
    cf_masking = Masking(mask_value = 0.0)(cf_input)
    cf_norm = BatchNormalization()(cf_masking)
    cf_lstm = Bidirectional(LSTM(10))(cf_norm) #change
    
    rf_input = Input(shape=(10,17), dtype='float32', name='rf_input')
    rf_masking = Masking(mask_value = 0.0)(rf_input)
    rf_norm = BatchNormalization()(rf_masking)
    rf_lstm = Bidirectional(LSTM(10))(rf_norm) #change
    
    rq_input = Input(shape=(20,50), dtype='float32', name='rq_input')
    rq_masking = Masking(mask_value = 0.0)(rq_input)
    rq_norm = BatchNormalization()(rq_masking)
    rq_lstm = Bidirectional(LSTM(30), name='rq_layer')(rq_norm) #change
    
    dj_input = Input(shape=(5,6), dtype='float32', name='dj_input')
    dj_masking = Masking(mask_value = 0.0)(dj_input)
    dj_norm = BatchNormalization()(dj_masking)
    dj_lstm = Bidirectional(LSTM(6))(dj_norm) #change
    
    wr_input = Input(shape=(4,), dtype='float32', name='wr_input')
    wr_norm = BatchNormalization()(wr_input)
    wr_relu = Dense(3, activation='relu')(wr_norm) #change
    
    cq_input = Input(shape=(10,48), dtype='float32', name='cq_input')
    cq_masking = Masking(mask_value = 0.0)(cq_input)
    cq_norm = BatchNormalization()(cq_masking)
    cq_lstm = Bidirectional(LSTM(30))(cq_norm) #change
    
    merged1 = concatenate([pd_relu, context_relu, cf_lstm, rf_lstm, rq_lstm, dj_lstm, wr_relu, cq_lstm])
    
    merged2 = Dense(32, activation='relu')(merged1) #change
    merged_dropout = Dropout(0.5)(merged2) #change
    merged3 = Dense(32, activation='relu', name = 'intermediary')(merged_dropout) #change
    merged_dropout2 = Dropout(0.5)(merged3) #change
    
    prob_good = Dense(1, activation='sigmoid')(merged_dropout2)
    
    model = Model(inputs=[pd_input, context_input, cf_input, rf_input, rq_input, dj_input, wr_input, cq_input], outputs=prob_good)
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0002, mode='min')
    
    model.fit([pd_train, context_train, cf_train, rf_train, rq_train, dj_train, wr_train, cq_train], gbf_train, validation_split=0.2, epochs=1000, batch_size=6400, callbacks=[early_stopping])
    
    #model.evaluate([pd_test, context_test, cf_test, rf_test, rq_test, dj_test], gbf_test, batch_size=50)
    
    yp = model.predict([pd_test, context_test, cf_test, rf_test, rq_test, dj_test, wr_test, cq_test], batch_size=1600)
    
    fpr, tpr, thresholds = metrics.roc_curve(gbf_test, yp.reshape(yp.size), pos_label = 1)
    print('Gini: ', metrics.auc(fpr, tpr) * 2 - 1)
    
    #run up to here
    
    
    def pg_to_score(x):
        return math.log(1/(1/x-1), 2)*100 + 200
    
    prob = list(map(lambda x: min(x, 0.9999),yp.reshape(yp.size)))
    score = np.array(list(map(pg_to_score, prob)))
    density = gaussian_kde(score)
    xs = np.arange(0,1200,1)
    #density.covariance_factor = lambda : .1
    #density._compute_covariance()
    plt.plot(xs,density(xs))
    plt.show()       
    
    np.min(score)
    np.max(score)
       
    #visualize with t-SNE
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('intermediary').output)
    intermediate_output = intermediate_layer_model.predict([pd_test, context_test, cf_test, rf_test, rq_test, dj_test, wr_test, cq_test])
    
    np.savetxt("../data/intermediate_output.csv", intermediate_output[range(100)], delimiter=",", fmt='%10.4f')
    np.savetxt("../data/gbf_test.csv", gbf_test[range(100)], delimiter=",", fmt='%1.0f')
    picked_rows = [gbf_test==0]
    intermediate_sample = intermediate_output[picked_rows][range(20)]
    gbf_sample = gbf_test[picked_rows][range(20)]
    np.savetxt("../data/intermediate_output2.csv", intermediate_sample, delimiter=",", fmt='%10.4f')
    np.savetxt("../data/gbf_test2.csv", gbf_sample, delimiter=",", fmt='%1.0f')
    
    