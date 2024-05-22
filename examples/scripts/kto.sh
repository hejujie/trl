EXP_NAME="ultra_interact_all_1.8b_single"

python3 examples/scripts/kto.py \
    --dataset_name="../datasets/openbmb/ultraInteract_pair" \
    --dataset_task_types="Logic" \
    --dataset_num_proc=50 \
    --model_name_or_path="/maindata/data/shared/public/yue.zhang/code/classifier_first/Qwen1p5-1p8b-Chat" \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 5e-7 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="../models/outputs/$EXP_NAME" \
    --warmup_ratio 0.1 \
    --warmup_steps 150 \
    --report_to wandb \
    --run_name $EXP_NAME \
    --bf16 \
    --logging_first_step