# Segmentation
Segmentation Research

# Eval Usage
python eval_seg_model.py --weight_url ckpts/b0.pt

# Training Usage
python train_seg_model.py path/to/config_file --path experiment_dir --weight_url ckpts/b0.pt


Config file
* Found in configs folder. The most important file for training that specifies all the details or training - # epochs, optimizer / weight decay, warmup epochs, learning rate, etc.

experiment_dir
*This is the folder that stores all the information for the experiment - the log of losses at each epoch, validation accuracy whenever it is calculated (this is typically every 25 epochs, but can be changed), checkpoint files for the model during training (saved every 25 epochs by default, but can be changed with --save_freq command line arg), etc.