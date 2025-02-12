export CUDA_VISIBLE_DEVICES="0" && python train.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --text_column maintext \
    --summary_column title \
    --max_source_length 256 \
    --max_target_length 64 \
    --num_beams 5 \
    --top_k 0 \
    --top_p 1.0 \
    --temperature 1.0 \
    --do_sample False \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-4 \
    --num_train_epochs 12 \
    --gradient_accumulation_steps 4 \
    --output_dir ./model
