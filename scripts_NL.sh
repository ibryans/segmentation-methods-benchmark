# NETHERLANDS F3 BLOCK EVALUATION

# DISCRETE EVALUATION
python section_train.py --architecture all --dataset NL --device cuda:0 --loss_function abl --n_epoch 60 --batch_size 16 --class_weights
python section_train.py --architecture all --dataset NL --device cuda:0 --loss_function abl --n_epoch 60 --batch_size 16 --class_weights --aug

# FILTER EVALUATION
python section_train.py --architecture all --dataset NL --device cuda:0 --loss_function abl --n_epoch 60 --batch_size 16 --class_weights --aug --filter canny 
python section_train.py --architecture all --dataset NL --device cuda:0 --loss_function abl --n_epoch 60 --batch_size 16 --class_weights --aug --filter gabor
python section_train.py --architecture all --dataset NL --device cuda:0 --loss_function abl --n_epoch 60 --batch_size 16 --class_weights --aug --filter sobel

# DELTA EVALUATION
python section_train.py --architecture all --dataset NL --device cuda:0 --loss_function abl --n_epoch 60 --batch_size 16 --class_weights --channel_delta 3 
# python section_train.py --architecture all --dataset NL --device cuda:0 --loss_function abl --n_epoch 60 --batch_size 16 --class_weights --aug --channel_delta 3 


# EVALUATION
python section_test.py --dataset NL --device cuda:0 
