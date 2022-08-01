import trainer as model
import torch

def main():
    print("starting training")
    model.trainer(instance_name = "Efficient_Net_B3_No_FP16")
    print("Training completed")
    torch.cuda.empty_cache()

if __name__ == '__main__':
     main()