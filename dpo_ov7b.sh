
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=en,eth,em,bond
export NCCL_DEBUG=INFO

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"



PROMPT_VERSION="qwen_1_5"



EPOCH=1
beta=0.1 



DPO_RUN_NAME="llava-onevision-qwen2-7b-ov_dpo-beta${beta}-epoch${EPOCH}"
DPO_CLEAN_NAME="${DPO_RUN_NAME##*/}"





SFT_MODEL="checkpoint-1284-trans"
SAVE_STEPS=183
FIRST_WRITE=False
TRAIN_BSZ_PER_DEVICE=4
OUTPUT_DIR="${DPO_CLEAN_NAME}"
DATA_PATH="jid2pairwise_data.llava_dpo_format_max_len_1003_filter.reidx.with_logps.pkl"
REF_LOGS_PATH="ref_logps.pkl"



echo $DPO_RUN_NAME

IMAGE_FOLDER='/'



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=22224 \
    llava/train/train_dpo.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path=${SFT_MODEL} \
    --dpo_alpha=1.0 \
    --beta=${beta} \
    --gamma=0 \
    --version $PROMPT_VERSION \
    --data_path=$DATA_PATH \
    --image_folder ${IMAGE_FOLDER} \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --unfreeze_mm_vision_tower False \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $DPO_CLEAN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size ${TRAIN_BSZ_PER_DEVICE} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 20 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --enable_removing_unnecessary_token True \
    --torch_empty_cache_steps 60 \
    --log_level info \
    --precompute_ref_log_probs True \
    --first_write ${FIRST_WRITE} \
    --ref_logps_save_path=${REF_LOGS_PATH} \
    2>&1 | tee log.log 


#    --save_steps 122 \ 