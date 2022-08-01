
import model
import gc


def main():
    print("started")
    model.runner(500)
    print("ended")


if __name__ == '__main__':
    # import torch
    # random_seed = 1 # or any of your favorite number 
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # import numpy as np
    # np.random.seed(random_seed)
    main()