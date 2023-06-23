# NEW ZEALAND PARIHAKA EVALUATION

# DISCRETE EVALUATION
python section_train.py --architecture all --dataset NZ --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --per_val 0.80
python section_train.py --architecture all --dataset NZ --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --per_val 0.80

# FILTER EVALUATION
python section_train.py --architecture all --dataset NZ --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --per_val 0.80 --filter canny
python section_train.py --architecture all --dataset NZ --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --per_val 0.80 --filter gabor
python section_train.py --architecture all --dataset NZ --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --per_val 0.80 --filter sobel

# VALIDATION SIZE EVALUATION
python section_train.py --architecture all --dataset NZ --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --per_val 0.97
python section_train.py --architecture all --dataset NZ --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --per_val 0.95
python section_train.py --architecture all --dataset NZ --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --per_val 0.90

# DELTA EVALUATION
python section_train.py --architecture all --dataset NZ --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --per_val 0.80 --channel_delta 3
# python section_train.py --architecture all --dataset NZ --device cuda:0 --n_epoch 100 --batch_size 10 --class_weights --aug --per_val 0.80 --channel_delta 3


# EVALUATION
python section_test.py --dataset NZ --device cuda:0 --split eval 
