import torch
import timm

def effi():
	model = timm.create_model("efficientnet_b0",
							num_classes = 2,
							drop_rate=0.3,
							drop_path_rate=0.2)
	return model