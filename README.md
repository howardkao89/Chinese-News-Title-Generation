# CSIE5431 Applied Deep Learning Homework 2
* Name: 高榮浩
* ID: R12922127

## Environment
* Ubuntu 20.04
* GeForce RTX™ 2080 Ti 11G
* Python 3.9
* CUDA 11.8

## Download
```sh
bash ./download.sh
```

The context for ```download.sh``` is as follows.

```sh
gdown --folder 1sk-FxBCQIVSQs1J6GZX9FKOagJ-yjmvz

```

## Training
```sh
bash ./train.sh
```

The context for ```train.sh``` is as follows.

```sh
export CUDA_VISIBLE_DEVICES="0" && python train.py \
    --model_name_or_path {...} \
    --train_file {...} \
    --validation_file {...} \
    --text_column {...} \
    --summary_column {...} \
    --max_source_length {...} \
    --max_target_length {...} \
    --num_beams {...} \
    --top_k {...} \
    --top_p {...} \
    --temperature {...} \
    --do_sample {...} \
    --per_device_train_batch_size {...} \
    --per_device_eval_batch_size {...} \
    --learning_rate {...} \
    --num_train_epochs {...} \
    --gradient_accumulation_steps {...} \
    --output_dir {...}

```

### Arguments
* ```model_name_or_path```: Path to model identifier from huggingface.co/models.
* ```train_file```: A jsonl file containing the training data.
* ```validation_file```: A jsonl file containing the validation data.
* ```text_column```: The name of the column in the datasets containing the full texts (for summarization).
* ```summary_column```: The name of the column in the datasets containing the summaries (for summarization).
* ```max_source_length```: The maximum total input sequence length after tokenization.
* ```max_target_length```: The maximum total sequence length for target text after tokenization.
* ```num_beams```: Number of beams for beam search. 1 means no beam search.
* ```top_k```: The number of highest probability vocabulary tokens to keep for top-k-filtering.
* ```top_p```: If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
* ```temperature```: The value used to modulate the next token probabilities.
* ```do_sample```: Whether or not to use sampling ; use greedy decoding otherwise.
* ```per_device_train_batch_size```: Batch size (per device) for the training dataloader.
* ```per_device_eval_batch_size```: Batch size (per device) for the evaluation dataloader.
* ```learning_rate```: Initial learning rate (after the potential warmup period) to use.
* ```num_train_epochs```: Total number of training epochs to perform.
* ```gradient_accumulation_steps```: Number of updates steps to accumulate before performing a backward/update pass.
* ```output_dir```: Where to store the final model.

The arguments I used are as shown in the following table.

| Argument | Value |
|:--------:|:-----:|
| model_name_or_path | google/mt5-small |
| train_file | ./data/train.jsonl |
| validation_file | ./data/public.jsonl |
| text_column | maintext |
| summary_column | title |
| max_source_length | 256 |
| max_target_length | 64 |
| num_beams | 5 |
| top_k | 0 |
| top_p | 1.0 |
| temperature | 1.0 |
| do_sample | False |
| per_device_train_batch_size | 4 |
| per_device_eval_batch_size | 4 |
| learning_rate | 5e-4 |
| num_train_epochs | 12 |
| gradient_accumulation_steps | 4 |
| output_dir | ./model |

## Prediction
```sh
bash ./pred.sh
```

The context for ```pred.sh``` is as follows.

```sh
export CUDA_VISIBLE_DEVICES="0" && python pred.py \
    --model_name_or_path {...} \
    --test_file {...} \
    --text_column {...} \
    --max_source_length {...} \
    --max_target_length {...} \
    --num_beams {...} \
    --top_k {...} \
    --top_p {...} \
    --temperature {...} \
    --do_sample {...} \
    --per_device_test_batch_size {...} \
    --output_path {...}

```

### Arguments
* ```model_name_or_path```: Path to pretrained model.
* ```test_file```: A jsonl file containing the testing data.
* ```text_column```: The name of the column in the datasets containing the full texts (for summarization).
* ```max_source_length```: The maximum total input sequence length after tokenization.
* ```max_target_length```: The maximum total sequence length for target text after tokenization.
* ```num_beams```: Number of beams for beam search. 1 means no beam search.
* ```top_k```: The number of highest probability vocabulary tokens to keep for top-k-filtering.
* ```top_p```: If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
* ```temperature```: The value used to modulate the next token probabilities.
* ```do_sample```: Whether or not to use sampling ; use greedy decoding otherwise.
* ```per_device_test_batch_size```: Batch size (per device) for the testing dataloader.
* ```output_path```: Where to store the final output.

The arguments I used are as shown in the following table.

| Argument | Value |
|:--------:|:-----:|
| model_name_or_path | ./model |
| test_file | ./data/public.jsonl |
| text_column | maintext |
| max_source_length | 256 |
| max_target_length | 64 |
| num_beams | 15 |
| top_k | 0 |
| top_p | 1.0 |
| temperature | 1.0 |
| do_sample | False |
| per_device_test_batch_size | 16 |
| output_path | ./pred/submission.jsonl |

## Prediction with Specified Path
```sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

The context for ```run.sh``` is as follows.

```sh
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

```

## Evaluation
```sh
python ./eval.py [-h] [-r REFERENCE] [-s SUBMISSION]
```

### Arguments
* ```-h, --help```: Show this help message and exit.
* ```-r REFERENCE, --reference REFERENCE```: Path to reference file.
* ```-s SUBMISSION, --submission SUBMISSION```: Path to submission file.

### Reference
[cccntu/tw_rouge](https://github.com/cccntu/tw_rouge)