export TRAIN_SHARDS_PATH_OR_URL="/data/code/songtao.tian/data/niji/image_info_20250616.json"
export PRETRAINED_TEACHER_MODEL="/data/code/guanyu.zhao/models/FLUX.1-dev"
export OUTPUT_DIR='outputs/TDD_flux_niji_random_0_3_proportion_empty_prompts_0.2'
# 
# 
# python  train_tdd_adv.py \
accelerate launch --config_file=config.yaml train_tdd_adv.py \
    --pretrained_teacher_model=$PRETRAINED_TEACHER_MODEL \
    --train_shards_path_or_url=$TRAIN_SHARDS_PATH_OR_URL \
    --output_dir=$OUTPUT_DIR \
    --seed=453645634 \
    --resolution=1024 \
    --max_train_samples=40000000 \
    --max_train_steps=1000000 \
    --train_batch_size=2 \
    --dataloader_num_workers=32 \
    --gradient_accumulation_steps=4 \
    --checkpointing_steps=100 \
    --validation_steps=50000000 \
    --learning_rate=5e-06 \
    --lora_rank=64 \
    --guidance_scale="random_0_3" \
    --mixed_precision="fp16" \
    --loss_type="huber"  --use_fix_crop_and_size --adam_weight_decay=0.0 \
    --val_infer_step=4 \
    --gradient_checkpointing --use_8bit_adam --set_grad_to_none \
    --num_euler_timesteps=250 \
    --proportion_empty_prompts=0.2 \
    --num_inference_steps_min=4 \
    --num_inference_steps_max=8 \
    --s_ratio=0.0 \
    --adv_lr=1e-5 \
    --adv_weight=0.1 \
    --resume_from_checkpoint=latest \

