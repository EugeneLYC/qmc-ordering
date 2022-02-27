model=resnet20
dataset=cifar100
epochs=200
bsz=128
lr=0.1
stype=SO

run_cmd="python3 main.py --model=${model} \
        --dataset=${dataset} \
        --epochs=${epochs} \
        --batch_size=${bsz} \
        --lr=${lr} \
        --shuffle_type=${stype} \
        --momentum=0.9 \
        --weight_decay=5e-4 \
        --num_classes=100 \
        --log_metric \
        --use_uniform_da
        "

cd ..

echo ${run_cmd}
eval ${run_cmd}