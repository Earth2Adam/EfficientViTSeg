from efficientvit.models.utils import load_state_dict_from_file
import os 
import torch

weight_url = 'experiments/rellis-lr3-n3/checkpoint/model_best_400.pt'

#model = create_seg_model('b0', 'rellis', weight_url=weight_url)
#model = torch.nn.DataParallel(model).cuda()
#model.eval()


file = os.path.realpath(os.path.expanduser(weight_url))
checkpoint = torch.load(file, map_location="cpu")
for k, v in checkpoint.items():
    print(k)

#weight = load_state_dict_from_file(weight_url)
#print(weight)
#model.load_state_dict(weight)