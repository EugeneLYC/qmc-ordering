TASK=stsb

base_dir=`pwd`

python run_glue_no_trainer.py \
  --model_name_or_path /home/ubuntu/data/transformer_ckpt/bertbase \
  --task_name ${TASK} \
  --train_file /home/ubuntu/data/GLUE/${TASK}/train.csv \
  --validation_file /home/ubuntu/data/GLUE/${TASK}/dev.csv \
  --max_length 128 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --use_tensorboard \
  --tensorboard_path ${base_dir}/text-classification \
  --seed $1 \
  --shuffle_type $2 \
  --zo_batch_size 8