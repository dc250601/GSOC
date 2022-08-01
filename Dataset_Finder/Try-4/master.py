import model
import gc


def main():
    parameters = [50,200,100,45,55,190,210,30,500] #500, 200, 100,  Are runned
    #8,6,4,3 will be runned separately

    # import torch
    # random_seed = 96 # or any of your favorite number 
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # import numpy as np
    # np.random.seed(random_seed)
    
    for parameter in parameters:
        import model
        import torch
        import torch
    
        print(f"\nprint from master.... parameter = {parameter}")
        model.runner(parameter)
        print("second ckpt")
        gc.collect()
        torch.cuda.empty_cache()
    

if __name__ == '__main__':
     main()
