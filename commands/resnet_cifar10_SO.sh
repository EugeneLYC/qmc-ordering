model=resnet20
dataset=cifar10
epochs=200
bsz=128
lr=0.1
stype=RR

JOB_NAME=stype_${stype}

run_cmd="python3 main.py --model=${model} \
        --dataset=${dataset} \
        --epochs=${epochs} \
        --batch_size=${bsz} \
        --lr=${lr} \
        --shuffle_type=${stype} \
        --momentum=0.9 \
        --weight_decay=1e-4 \
        --use_uniform_da \
        --use_qr_decomposition_sort \
        --use_fuse_tensor
        &> ${JOB_NAME}.log
        "

cd ..

echo ${run_cmd}
eval ${run_cmd}