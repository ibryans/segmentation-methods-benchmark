# NOVA SCOTIA PENOBSCOT EVALUATION

# DISCRETE EVALUATION
python section_train.py --architecture all --dataset NS --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights
python section_train.py --architecture all --dataset NS --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug

# FILTER EVALUATION
python section_train.py --architecture all --dataset NS --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --filter canny
python section_train.py --architecture all --dataset NS --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --filter gabor 
python section_train.py --architecture all --dataset NS --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --filter sobel

# DELTA EVALUATION
python section_train.py --architecture all --dataset NS --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --channel_delta 3
# python section_train.py --architecture all --dataset NS --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --channel_delta 3


# EVALUATION
python section_test.py --dataset NS --device cuda:0 
