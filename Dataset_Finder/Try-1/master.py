import model
import gc


def main():
    parameters = [500] #3 ,4, 6, 8, 10, 15, 20, 30, 50, 100, 200 Are runned

    for parameter in parameters:
        import model
        import torch
        print(f"\nprint from master.... parameter = {parameter}")
        model.runner(parameter)
        print("second ckpt")
        gc.collect()
        torch.cuda.empty_cache()
    

if __name__ == '__main__':
     main()
