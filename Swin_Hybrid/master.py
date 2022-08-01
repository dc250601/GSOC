import hybrid_trainer as model
import torch

def main():
    print("starting training")
    model.hybrid_trainer(instance_name = "Hybrid_base")
    print("Training completed")
    torch.cuda.empty_cache()

if __name__ == '__main__':
     main()