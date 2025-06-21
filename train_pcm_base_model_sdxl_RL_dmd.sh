PREFIX=PCM_sdxl_base_model_RL_laion_dmd2
MODEL_DIR=/aisocial/shejiao/songtao.tian/models/stable-diffusion-xl-base-1.0
VAE_DIR=/aisocial/shejiao/songtao.tian/models/sdxl-vae-fp16-fix
DATA_DIR=/data/code/songtao.tian/data/data/cc3m/image_info.json
RL_EPSILON=0.5
DMD_WEIGHT=0.3
OUTPUT_DIR="outputs/base_model_${PREFIX}_RL_epsilon_${RL_EPSILON//./_}_dmd_${DMD_WEIGHT//./_}"
PROJ_NAME="base_model_$PREFIX"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOGFILE="${PREFIX}_${TIMESTAMP}.log"

# 
# 
# 
# python train_pcm_base_model_sdxl_adv_RL.py \
accelerate launch --main_process_port 29501 train_pcm_base_model_sdxl_adv_RL.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --pretrained_vae_model_name_or_path=$VAE_DIR \
    --output_dir=$OUTPUT_DIR \
    --tracker_project_name=$PROJ_NAME \
    --train_shards_path_or_url=$DATA_DIR \
    --mixed_precision=fp16 \
    --resolution=614 \
    --lora_rank=64 \
    --learning_rate=1e-6 --loss_type="huber" --adam_weight_decay=0 \
    --max_train_steps=200000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=0 \
    --w_min=6 \
    --w_max=7 \
    --validation_steps=10000000 \
    --checkpointing_steps=250 --checkpoints_total_limit=10000 \
    --train_batch_size=4 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=8 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=20250410 \
    --num_ddim_timesteps=40 \
    --multiphase=4 \
    --adv_lr=1e-5 \
    --allow_tf32 \
    --adv_weight=0.1 \
    --gradient_checkpointing \
    --ema_decay=0.99 \
    --not_use_crop \
    --use_fp16 \
    --sdxl \
    --RL_epsilon=$RL_EPSILON \
    --dmd_weight=$DMD_WEIGHT \
    --dmd_loss
# Reduce batch size if GPU memory is limited
# tf32 for slightly faster training
# 2k iterations is enough to see clear improvements.
