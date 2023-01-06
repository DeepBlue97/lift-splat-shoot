NUSCENES_ROOT=/home/YQ_Wang/ocean/datasets/nuscenes
NUSCENES_ROOT=/media/peter/ocean/data/dataset/nuscenes

# python main.py train trainval --dataroot=$NUSCENES_ROOT --logdir=./runs --gpuid=0
# tensorboard --logdir=./runs --bind_all

python main.py train mini --dataroot=$NUSCENES_ROOT --logdir=./runs --gpuid=0
tensorboard --logdir=./runs --bind_all

# python main.py train mini/trainval --dataroot=$NUSCENES_ROOT --logdir=./runs --gpuid=0
# tensorboard --logdir=./runs --bind_all

# python main.py train mini/trainval --dataroot=NUSCENES_ROOT --logdir=./runs --gpuid=0
# tensorboard --logdir=./runs --bind_all

