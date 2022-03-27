model=logistic_regression
dataset=mnist
epochs=100
bsz=64
lr=0.01
stype=fresh_grad_greedy_sort

base_dir=`pwd`

run_cmd="python train_logreg_mnist.py --model=${model} \
        --dataset=${dataset} \
        --data_path=${base_dir} \
        --epochs=${epochs} \
        --batch_size=${bsz} \
        --lr=${lr} \
        --shuffle_type=${stype} \
        --momentum=0.9 \
        --weight_decay=1e-4 \
        --log_metric \
        --use_tensorboard \
        --tensorboard_path=${base_dir} \
        --num_workers=0 \
        --transforms_json=${base_dir}/jsons/mnist.json
        "

echo ${run_cmd}
eval ${run_cmd}
