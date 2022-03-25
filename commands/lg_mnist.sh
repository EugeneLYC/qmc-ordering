model=logistic_regression
dataset=mnist
epochs=100
bsz=64
lr=0.01
stype=fresh

base_dir=`pwd`

run_cmd="python main.py --model=${model} \
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
        --workers=0
        "

echo ${run_cmd}
eval ${run_cmd}

        # --use_uniform_da \
        # --use_random_proj \
        # --zo_batch_size=8 \
        # --proj_target=8