from tkinter.tix import MAIN
import numpy as np
from pip import main
path_test = "Test/"
path_train = "Train/"
import cv2
import pyarrow.parquet as pq
import gc
import time as time
def generate(pf, path, threshold, ab, parameter):
    record_batch = pf.iter_batches(batch_size=4*1024)
    while True:
        try:

            batch = next(record_batch)
            ab = transform(batch, path, parameter, threshold, ab)

        except StopIteration:
            return ab

def transform(batch, path, parameter, threshold, ab):
    p = batch.to_pandas()
    im = np.array(np.array(np.array(p.iloc[:,0].tolist()).tolist()).tolist())
    meta = np.array(p.iloc[:,3])
    return saver(im,meta, path, parameter, threshold, ab)


def saver(im, meta, path, parameter, threshold, ab):
    alpha, beta = ab

    im[im < threshold] = 0 #Zero_suppression
    im[:,0,:,:] = (im[:,0,:,:] - im[:,0,:,:].mean())/(im[:,0,:,:].std())
    im[:,1,:,:] = (im[:,1,:,:] - im[:,1,:,:].mean())/(im[:,1,:,:].std())
    im[:,2,:,:] = (im[:,2,:,:] - im[:,2,:,:].mean())/(im[:,2,:,:].std())


    # im[:,0,:,:] = np.clip(im[:,0,:,:], a_min = 0, a_max = 10*im[:,0,:,:].std(axis = (1,2))[:,None,None])
    # im[:,1,:,:] = np.clip(im[:,1,:,:], a_min = 0, a_max = 10*im[:,1,:,:].std(axis = (1,2))[:,None,None])
    # im[:,2,:,:] = np.clip(im[:,2,:,:], a_min = 0, a_max = 10*im[:,2,:,:].std(axis = (1,2))[:,None,None])



    for i in range(meta.shape[0]):
        img = im[i, :, :, :]

        channel1 = img[0, :, :]
        channel2 = img[1, :, :]
        channel3 = img[2, :, :]


        channel1 = np.clip(channel1, 0, parameter*channel1.std())
        channel2 = np.clip(channel2, 0, parameter*channel2.std())
        channel3 = np.clip(channel3, 0, parameter*channel3.std())


        channel1 = 255*(channel1)/(channel1.max()) #The normalisation is such because the data is already
        channel2 = 255*(channel2)/(channel2.max()) #clipped
        channel3 = 255*(channel3)/(channel3.max())


        img[0,:,:] = channel1
        img[1,:,:] = channel2
        img[2,:,:] = channel3


        img = img.astype(np.uint8)
        img = img.T

        if(meta[i] == 0):
            impath = path+"1/"+str(alpha)+".png"
            alpha = alpha + 1
        if(meta[i] == 1):
            impath = path+"0/"+str(beta)+".png"
            beta = beta + 1
        cv2.imwrite(impath , img)


    return [alpha, beta]


def runner(threshold, parameter = 50):
    print(f"Entering Runner threshold is {threshold} and parameter is {parameter}")
    ab = [0, 0]
    ab = generate(pq.ParquetFile("./Data/test0.parquet"), path_test, threshold=threshold, ab=ab, parameter=parameter)
    ab = generate(pq.ParquetFile("./Data/test1.parquet"), path_test, threshold=threshold, ab=ab, parameter=parameter)
    ab = generate(pq.ParquetFile("./Data/test2.parquet"), path_test, threshold=threshold, ab=ab, parameter=parameter)
