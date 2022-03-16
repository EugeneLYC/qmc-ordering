model=resnet18
dataset=imagenet
epochs=5
bsz=256
lr=1e-4
stype=RR

base_dir=`pwd`

run_cmd="python3 main.py --model=${model} \
        --dataset=${dataset} \
        --data_path=/data/imagenet/ \
        --epochs=${epochs} \
        --batch_size=${bsz} \
        --lr=${lr} \
        --shuffle_type=${stype} \
        --momentum=0.9 \
        --weight_decay=1e-4 \
        --use_tensorboard \
        --tensorboard_path=${base_dir} \
        --workers=16 \
        --use_qmc_da \
        --transforms_json=${base_dir}/jsons/imagenet.json
        "

echo ${run_cmd}
eval ${run_cmd}
