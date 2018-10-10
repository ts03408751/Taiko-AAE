
# coding: utf-8

import DTW_song_dongKa_sampling as DTW
import matplotlib.pyplot as plt
import gc
import numpy as np

#查詢data organize table (母)後一個一個執行 DTW_song_dongKa_sampling並且串成train set

Dir = 'motif/aaaaa/song1/order1/don'
sample_order1 = DTW.Main_Execure(Dir)
train_set = sample_order1

#AAE

def create_lstm_vae_train(input_dim, 
timesteps, 
batch_size, 
intermediate_dim, 
latent_dim,
epsilon_std=1.,
first_second=False                                
):

    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator. 

    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM. 
        latent_dim: int, latent z-layer shape. 
        epsilon_std: float, z-layer sigma.


    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(timesteps, input_dim,))
    if (first_second==False):
        # LSTM encoding
        h = LSTM(intermediate_dim)(x)

        # VAE Z layer
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)
    elif(first_second==True): #fix weights for encoder
        # LSTM encoding
        h = LSTM(intermediate_dim)(x)

        # VAE Z layer
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    if(first_second ==True):
        encoder.load_weights('encoder.h5')

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    return vae, encoder, generator

# training
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from keras.constraints import non_neg
from sklearn.preprocessing import MinMaxScaler
from lstm_vae import create_lstm_vae
from keras.callbacks import History 
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, RepeatVector
from keras.backend.tensorflow_backend import set_session

def conservative_train_AAE(data1,data2=[]):
    
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config)) 

    #traning
    if __name__ == "__main__":
        x = data1
        y = data2
        
        
        
        if(y!=[]):
            consercative_or_not = True
        else:
            consercative_or_not = False
        
        if (consercative_or_not ==True):
            xY = np.vstack((x,y))
            
            input_dim = xY.shape[-1] 
            print (input_dim)
            timesteps = xY.shape[1]
            print (timesteps)
            batch_size = 1
          
            
            vae, enc, gen = create_lstm_vae_train(input_dim, 
            timesteps, 
            batch_size, 
            intermediate_dim=64,
            latent_dim=5,
            epsilon_std=1.,
            first_second= True
                                )
            
            early_stopping = EarlyStopping(monitor='loss', patience=100, verbose=2)
            vae.fit(xY, xY, epochs=10000,callbacks=[early_stopping]) 
            enc.save_weights('encoder.h5')
            preds = vae.predict(xY,batch_size=batch_size)
            
        else:
            input_dim = x.shape[-1] # 13
            print (input_dim)
            timesteps = x.shape[1] # 3
            print (timesteps)
            batch_size = 1
            
       
            vae, enc, gen = create_lstm_vae_train(input_dim, 
            timesteps, 
            batch_size, 
            intermediate_dim=64,
            latent_dim=5,
            epsilon_std=1.)

            early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=2)
            vae.fit(x, x, epochs=10000,callbacks=[early_stopping])
            enc.save_weights('encoder.h5')
            preds = vae.predict(x,batch_size=batch_size)
    return vae, enc, gen

    gc.collect()

#查詢data organize table (tmp)後一個一個執行 DTW_song_dongKa_sampling並且串成train set

#train(母)串 train(子), then train
vae, enc, gen  = conservative_train_AAE(data1 = train_set)
print (' train data finish!')
gc.collect()

#雷達圖 
def Rader(sample):
    Latent_code = enc.predict(sample)
    samplesrar = np.interp(Latent_code, (Latent_code.min(), Latent_code.max()), (50, 100)) #Rescale to 0-100
# 中文和負號的正常顯示
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 使用ggplot的繪圖風格
    plt.style.use('ggplot')

    # 構造數據
    values = samplesrar[0]



    feature = ['feature01','feature02','feature03','feature04','feature05']

    N = len(values)
    # 設置雷達圖的角度，用於平分切開一個圓面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)
    # 為了使雷達圖一圈封閉起來，需要下面的步驟
    values=np.concatenate((values,[values[0]]))

    angles=np.concatenate((angles,[angles[0]]))

    # 繪圖
    fig=plt.figure()
    ax = fig.add_subplot(111, polar=True)
    # 繪製折線圖
    ax.plot(angles, values, 'o-', linewidth=0.2, label = 'sample')
    # 填充顏色
    ax.fill(angles, values, alpha=0.25)
    # 繪製第二條折線圖


    # 添加每個特徵的標籤
    ax.set_thetagrids(angles * 180/np.pi, feature)
    # 設置雷達圖的範圍
    ax.set_ylim(0,100)
    # 添加標題
    plt.title('Radar_subject 02')

    # 添加網格線
    ax.grid(True)
    # 設置圖例
    plt.legend(loc = 'best')
    # 顯示圖形
    plt.savefig('Radar.png')
    plt.show()

Rader(sample_order1)

