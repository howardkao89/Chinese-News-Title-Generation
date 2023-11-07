export CUDA_VISIBLE_DEVICES="0" && python pred.py \
    --model_name_or_path ./model \
    --test_file ${1} \
    --text_column maintext \
    --max_source_length 256 \
    --max_target_length 64 \
    --num_beams 15 \
    --top_k 0 \
    --top_p 1.0 \
    --temperature 1.0 \
    --do_sample False \
    --per_device_test_batch_size 16 \
    --output_path ${2}
