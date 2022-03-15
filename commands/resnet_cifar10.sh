model=resnet20
dataset=cifar10
epochs=200
bsz=128
lr=0.1
stype=RR

base_dir=`pwd`

run_cmd="python3 main.py --model=${model} \
        --dataset=${dataset} \
        --data_path=${base_dir} \
        --epochs=${epochs} \
        --batch_size=${bsz} \
        --lr=${lr} \
        --shuffle_type=${stype} \
        --momentum=0.9 \
        --weight_decay=1e-4 \
        --use_tensorboard \
        --tensorboard_path=${base_dir} \
        --workers=4 \
        --use_qmc_da \
        --transforms_json=${base_dir}/jsons/resnet_cifar10.json
        "

echo ${run_cmd}
eval ${run_cmd}
