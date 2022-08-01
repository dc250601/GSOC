from tkinter.tix import MAIN
import numpy as np
from pip import main
path_test = "Test/"
path_train = "Train/"
import cv2
import pyarrow.parquet as pq
import gc

def generate(pf, path, parameter, ab):
    record_batch = pf.iter_batches(batch_size=1024)
    while True:
        try:
            batch = next(record_batch)
            # print(batch.num_rows)
            ab = transform(batch, path, parameter, ab)
        except StopIteration:
            # print("Done")
            return ab

def transform(batch, path, parameter, ab):
    p = batch.to_pandas()
    im = np.array(np.array(np.array(p.iloc[:,0].tolist()).tolist()).tolist())
    meta = np.array(p.iloc[:,3])
    return saver(im,meta, path, parameter, ab)

def saver(im, meta, path, parameter, ab):
    alpha, beta = ab
    channel1 = im[:,0,:,:]
    channel2 = im[:,1,:,:]
    channel3 = im[:,2,:,:]

    channel1 = np.clip(channel1, 0, parameter*channel1.std()) # 4 here is an hyper-parameter which need tuning
    channel2 = np.clip(channel2, 0, parameter*channel2.std())
    channel3 = np.clip(channel3, 0, parameter*channel3.std())
    
    channel1 = 255*(channel1)/(channel1.max()) #The normalisation is such because the data is already
    channel2 = 255*(channel2)/(channel2.max()) #clipped
    channel3 = 255*(channel3)/(channel3.max())
    
    im[:,0,:,:] = channel1
    im[:,1,:,:] = channel2
    im[:,2,:,:] = channel3
    
    im = im.astype(np.uint8)
    
    del channel1
    del channel2
    del channel3
    
    gc.collect()
    
    for i in range(meta.shape[0]):
        img = im[i,:,:,:]
        img = img.T
        if(meta[i] == 0):
            impath = path+"1/"+str(alpha)+".png"
            alpha = alpha + 1
        if(meta[i] == 1):
            impath = path+"0/"+str(beta)+".png"
            beta = beta + 1
        #lmm.save(impath)
        cv2.imwrite(impath , img)
    return [alpha, beta]

def runner(parameter):
    ab = [0, 0]
    ab = generate(pq.ParquetFile("./Data/test0.parquet"), path_test, parameter, ab)
    ab = generate(pq.ParquetFile("./Data/test1.parquet"), path_test, parameter, ab)
    ab = generate(pq.ParquetFile("./Data/test2.parquet"), path_test, parameter, ab)

