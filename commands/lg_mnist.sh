model=logistic_regression
dataset=mnist
epochs=100
bsz=128
lr=0.01
stype=ZO

run_cmd="python3 main.py --model=${model} \
        --dataset=${dataset} \
        --epochs=${epochs} \
        --batch_size=${bsz} \
        --lr=${lr} \
        --shuffle_type=${stype} \
        --momentum=0.9 \
        --weight_decay=1e-4 \
        --log_metric \
        --use_tensorboard \
        --zo_batch_size=1
        "

echo ${run_cmd}
eval ${run_cmd}