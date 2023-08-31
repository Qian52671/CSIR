from sklearn.metrics import mean_squared_error
from CAE_model import *
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import tensorflow.keras as keras



def CAE_evaluation(x,autoencoder):
    autoencoder.load_weights('./model_saver/model.h5')
    reconstructed_data = autoencoder.predict(x)

    rmse_list = []
    mae_list = []

    for i in range(reconstructed_data.shape[0]):
         original_sample = x[i].flatten() 
         reconstructed_sample = reconstructed_data[i].flatten() 
         rmse = np.sqrt(mean_squared_error(original_sample, reconstructed_sample))
         mae = np.mean(np.abs(original_sample - reconstructed_sample))

         rmse_list.append(rmse)
         mae_list.append(mae)

    average_rmse = np.mean(rmse_list)
    average_mae = np.mean(mae_list)


    print("Average RMSE:", average_rmse)
    print("Average MAE):", average_mae)


def flattened_images(x):
    flattened_images = []
    for i in range(len(x)):
        flattened = x[i].reshape(-1)
        flattened_images.append(flattened)
    flattened_images = np.array(flattened_images)
    print(flattened_images.shape)
    return flattened_images 



def PCA_evaluation(x):
    flattened_images = flattened_images(x) 
    pca = PCA()
    pca.fit(flattened_images)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    first_point_above_95 = np.argmax(cumulative_variance > 0.95)

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_variance, marker='o')
    plt.axvline(x=first_point_above_95, color='g', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(first_point_above_95 + 1, 0.6, f'First point above 0.95: {first_point_above_95}', color='g')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance vs. Number of Principal Components')
    plt.grid(True)
    plt.show()
    print("First point above 0.95:", first_point_above_95)

    pca = PCA(n_components=first_point_above_95)
    pca.fit(flattened_images)
    reduced_data = pca.transform(flattened_images)
    reconstructed_data = pca.inverse_transform(reduced_data)
    print(reconstructed_data.shape)

    rmse = np.sqrt(mean_squared_error(flattened_images, reconstructed_data))
    print("Root Mean Squared Error:", rmse)

    mae = np.mean(np.abs(flattened_images - reconstructed_data))
    print("Mean Absolute Error (MAE):", mae)

    return first_point_above_95




def KPCA_evaluation(x):
    flattened_images = flattened_images(x) 
    d = PCA_evaluation(x) 
    kpca = KernelPCA(n_components=d,fit_inverse_transform=True,kernel='rbf')
    kpca = kpca.fit(flattened_images)
    kpca_reduced_data = kpca.transform(flattened_images)
    reconstructed_data = kpca.inverse_transform(kpca_reduced_data)
    rmse = np.sqrt(mean_squared_error(flattened_images, reconstructed_data))
    print("Root Mean Squared Error:", rmse)

    mae = np.mean(np.abs(flattened_images - reconstructed_data))
    print("Mean Absolute Error (MAE):", mae)



pretrain_epochs = 100
batch_size = 40

def AE_model(train):
    input_dim = train.shape[1]
    print(input_dim)
    input = keras.Input(shape=(input_dim,))
    print(input)
    encoded = layers.Dense(500, activation="relu")(input)
    encoded = layers.Dense(500, activation='relu')(encoded)
    encoded = layers.Dense(2000, activation='relu')(encoded)
    encoded = layers.Dense(100, activation='sigmoid')(encoded)

    decoded = layers.Dense(2000, activation='relu')(encoded)
    decoded = layers.Dense(500, activation='relu')(decoded)
    decoded = layers.Dense(500, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation="relu")(decoded) 

    # use encoder to encode raw data
    return Model(inputs=input, outputs=decoded, name='AE'), Model(inputs=input, outputs=encoded, name='encoder')



def AE_evaluation(x):
    flattened_images = flattened_images(x) 
    AE, AE_encoder = AE_model(flattened_images)
    save_dir = './AE_model'
    AE.compile(optimizer='adam', loss='mse')
    AE.fit(flattened_images, flattened_images, batch_size=batch_size, epochs=pretrain_epochs)
    AE.save_weights(save_dir+'/conv_ae_weights.h5')
    reconstructed_data = AE.predict(flattened_images)
    reconstructed_data.shape
    rmse = np.sqrt(mean_squared_error(flattened_images, reconstructed_data))
    print("Root Mean Squared Error:", rmse)
    mae = np.mean(np.abs(flattened_images - reconstructed_data))
    print("Mean Absolute Error (MAE):", mae)

