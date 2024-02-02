import "submitters/scslurm.tape"

task DumpHFDataset 
    > dataset_json=dataset.json
    > dataset_json_test=dataset_test.json
    > test_data
    :: dataset_name=@
    :: dataset_path=@
    :: dataset_dirs=@
    :: dataset_stream=@
    :: filter=@
    :: percentile=@
    :: n_tokens=@
    :: n_tokens_test=@
    :: repo=@
    :: hf_dataset=@
    :: bilingual=@
    :: code=@
    :: pre_tokenizer=@
    :: .submitter=@
    :: .account=$dump_account
    :: .partition=$dump_partition
    :: .qos=$dump_qos
    :: .time=$dump_time 
    :: .cpus=$dump_cpus
    :: .mem=$dump_mem
{
    python $repo/scripts/dump_hf_dataset.py \
        --dataset_name $dataset_name \
        --output $dataset_json \
        --output_test $dataset_json_test \
        $([ ! -z "$dataset_dirs" ] && echo "--dataset_dirs $dataset_dirs" || echo "") \
        $([ ! -z "$dataset_path" ] && echo "--dataset_path $dataset_path" || echo "") \
        $([ "$filter" == true ] && echo "--filter" || echo "") \
        $([ ! -z "$percentile" ] && echo "--percentile $percentile" || echo "") \
        $([ ! -z "$n_tokens" ] && echo "--n_tokens $n_tokens" || echo "") \
        $([ "$dataset_stream" == true ] && echo "--stream" || echo "") \
        $([ "$hf_dataset" == true ] && echo "--hf_dataset" || echo "") \
        $([ "$bilingual" == true ] && echo "--bilingual" || echo "") \
        $([ "$code" == true ] && echo "--code" || echo "") \
        $([ ! -z "$n_tokens_test" ] && echo "--n_tokens_test $n_tokens_test" || echo "") \
        $([ ! -z "$pre_tokenizer" ] && echo "--pre_tokenizer $pre_tokenizer" || echo "") 

    # extract test data from json file (one per line)
    cat $dataset_json_test | jq -r '.text' > $test_data
}

# Warning: This is working, but isnt passed to subsequent tasks
task TrainTokenizer
    < dataset_json=@DumpHFDataset[Dataset:*]
    > tokenizer
    :: repo=@
    :: pre_tokenizer=$pre_tokenizer[Dataset:*]
    :: vocab_size=@
    :: words_per_source=$tokenizer_words_per_source
    :: extra_tokens=@
    :: .submitter=@
    :: .account=$traintok_account
    :: .partition=$traintok_partition
    :: .time=$traintok_time
    :: .cpus=$traintok_cpus
    :: .mem=$traintok_mem
{
    mkdir tok_corpus
    mkdir tokenizer
    echo "Preparing tokenizer corpus..."
    python $repo/scripts/prepare_tokenizer_corpus.py \
        --data_paths $dataset_json \
        --words_per_source $words_per_source \
        --output tok_corpus/data \
        --pre_tokenizer $pre_tokenizer
    echo "Training tokenizer..."
    python $repo/scripts/train_tokenizer.py \
        --data_path tok_corpus \
        --vocab_size $vocab_size \
        --extra_tokens $extra_tokens \
        --output $tokenizer
    #echo "Analyzing tokenizer..."
    #python $repo/scripts/analyze_tokenizer.py \
    #    --tokenizer_dir $tokenizer \
    #    --eval_sets $eval_sets \
}

task PreprocessDataset
    < dataset_json=@DumpHFDataset
    > dataset_bin=data_bin
    :: repo=@
    :: external_tokenizer=@
    :: cpu_workers=@
    :: .submitter=@
    :: .account=$preproc_account
    :: .partition=$preproc_partition
    :: .time=$preproc_time
    :: .cpus=$preproc_cpus
    :: .mem=$preproc_mem
{
    set -euo pipefail
    mkdir -p $dataset_bin
    # automatically get key from json file
    # json_key=$(head -n 1 $dataset_json | jq -r 'keys[0]')
    json_key="text"
    python $repo/Megatron-DeepSpeed/tools/preprocess_data.py \
        --input $dataset_json \
        --json-keys $json_key \
        --output-prefix $dataset_bin/data \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $external_tokenizer \
        --append-eod \
        --workers $cpu_workers

    # if key is different from `text`, rename bin files to `data_text_document.*`
    if [ "$json_key" != "text" ]; then
        mv $dataset_bin/data_${json_key}_document.bin $dataset_bin/data_text_document.bin
        mv $dataset_bin/data_${json_key}_document.idx $dataset_bin/data_text_document.idx
    fi
}

task PreprocessValidDataset
    < dataset_json_test=@DumpHFDataset
    > val_dataset_bin=val_data_bin
    :: repo=@
    :: external_tokenizer=@
    :: cpu_workers=20
    :: .submitter=@
    :: .account=$preproc_account
    :: .partition=$preproc_partition
    :: .time=$preproc_time
    :: .cpus=$preproc_cpus
    :: .mem=$preproc_mem
{
    set -euo pipefail
    mkdir -p $val_dataset_bin
    # automatically get key from json file
    # json_key=$(head -n 1 $dataset_json_test | jq -r 'keys[0]')
    # NOTE: for croissant, hardcode key to `text` 
    json_key="text"
    python $repo/Megatron-DeepSpeed/tools/preprocess_data.py \
        --input $dataset_json_test \
        --json-keys $json_key \
        --output-prefix $val_dataset_bin/data \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $external_tokenizer \
        --append-eod \
        --workers $cpu_workers

    # if key is different from `text`, rename bin files to `data_text_document.*`
    if [ "$json_key" != "text" ]; then
        mv $val_dataset_bin/data_${json_key}_document.bin $val_dataset_bin/data_text_document.bin
        mv $val_dataset_bin/data_${json_key}_document.idx $val_dataset_bin/data_text_document.idx
    fi
}

# WARNING: has not been tested recently
task GetDeepSpeedConfig
    < dataset_bin=@PreprocessDataset
    > ds_config=deepspeed.json
    :: repo=@
    :: nodes=@
    :: gpus=@
    :: zero_stage=@
    :: batch_size=@
    :: grad_accum_steps=@
{
    micro_batch_size=$(($batch_size / ($gpus * $grad_accum_steps * $nodes)))
    echo -n '{
        "train_batch_size" : '$batch_size',
        "train_micro_batch_size_per_gpu": '$micro_batch_size',
        "gradient_accumulation_steps": '$grad_accum_steps',
        "steps_per_print": 10,
        "zero_optimization": {
            "stage": '$zero_stage'
        },
        "bf16": {
            "enabled": true
        }
    }' | jq . > $ds_config
}

task GetValidPaths
    < val_dataset_bin=@PreprocessValidDataset
    > valid_data_file
    :: dataset_name=@
{
    echo "$dataset_name $val_dataset_bin/data_text_document" > $valid_data_file
}

task MakeDataMix
    < dataset_bin=@PreprocessDataset
    > datamix_file
    :: datamix_weights=@
{
    # simply write datamix weight and path in dataset_bin to a file, separated by a space
    echo "$datamix_weights $dataset_bin/data_text_document" > $datamix_file
}

task Train
    < datamix_file=@MakeDataMix[Dataset:*]
    < valid_data_file=@GetValidPaths[Dataset:*]
    < ds_config=@GetDeepSpeedConfig
    > model_dir=checkpoints
    :: .submitter=@
    :: .C=$train_C
    :: .account=$train_account
    :: .partition=$train_partition
    :: .reservation=$train_reservation
    :: .qos=$train_qos
    :: .time=$train_time
    :: .cpus=$train_cpus
    :: .nnodes=$train_nodes
    :: .gres=$train_gres
    :: .mem=$train_mem
    :: .restart_on_timeout=true
    :: repo=@
    :: external_model_dir=@
    :: external_resume=@
    :: external_tensorboard=@
    :: external_tokenizer=@
    :: nodes=@
    :: gpus=@
    :: tp=@
    :: pp=@
    :: master_port=@
    :: zero_stage=@
    :: model_config=@
    :: train_steps=@
    :: batch_size=@
    :: grad_accum_steps=@
    :: lr=@
    :: min_lr=@
    :: lr_warmup_steps=@
    :: weight_decay=@
    :: grad_clip=@
    :: save_interval=@
    :: eval_interval=@
    :: seed=@
    :: multiple_valid_sets=@
{
    set -x 

    data_path=""
    for f in $datamix_file; do
        # read file
        data_path="$data_path `cat $f`"
    done
    echo "Running with data_path=$data_path"

    if [ "$multiple_valid_sets" == true ]; then
        valid_data_path=""
        for f in $valid_data_file; do
            # read file
            valid_data_path="$valid_data_path `cat $f`"
        done
        echo "Running with valid_data_path=$valid_data_path"
    fi

    # check if number folders passed in dataset_bin (space separated) is equal to 
    # the number of datamix weights passed (space separated)
    # n_weights=$(echo $datamix_weights | tr ' ' '\n' | wc -l)
    # n_folders=$(echo $dataset_bin | tr ' ' '\n' | wc -l)
    # if [ "$n_weights" != "$n_folders" ]; then
    #     echo "ERROR: number of datamix weights ($n_weights) is not equal to number of dataset folders ($n_folders)"
    #     exit 1
    # fi

    # # make dataset_path for mix for megatron
    # weights=($datamix_weights)
    # folders=($dataset_bin)
    # data_path=""
    # # iterate over weights and folders
    # i=0
    # while [ $i -lt ${#weights[@]} ]
    # do
    #     data_path+="${weights[$i]} ${folders[$i]}/data_text_document "
    #     i=$((i+1))
    # done

    # Read model config and parse to variables
    model_config_f="${repo}/configs/models/${model_config}.yml"
    hidden_size=$(yq '.hidden_size' $model_config_f)
    ffn_hidden_size=$(yq '.ffn_hidden_size' $model_config_f)
    num_layers=$(yq '.num_layers' $model_config_f)
    num_attention_heads=$(yq '.num_attention_heads' $model_config_f)
    seq_length=$(yq '.seq_length' $model_config_f)
    num_kv_heads=$(yq '.num_kv_heads' $model_config_f)

    # if num_nodes is bigger than 1, get master_addr from the first node
    if [ "$nodes" -gt 1 ]; then
        master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    else
        master_addr="localhost"
    fi

    # read main optimization parameters from the DeepSpeed config file
    # (in case these were automatically tuned)
    zero_stage=$(jq -r '.zero_optimization.stage' $ds_config)
    micro_batch_size=$(jq -r '.train_micro_batch_size_per_gpu' $ds_config)
        
    ds_args="--deepspeed"
    ds_args="--deepspeed_config=$ds_config ${ds_args}"
    ds_args="--zero-stage=$zero_stage ${ds_args}"
    ds_args="--deepspeed-activation-checkpointing ${ds_args}"
    
    launcher="python -u -m torch.distributed.run"
    distributed_args="--nnodes=$nodes --nproc_per_node=$gpus "
    distributed_args="${distributed_args} --rdzv_backend c10d --rdzv_endpoint $master_addr:$master_port"
    echo "Using DeepSpeed with zero stage $zero_stage and micro batch size $micro_batch_size"

    distributed_args="$distributed_args"
    # if nodes is bigger than 1, add node_rank to ds_args
    if [ "$nodes" -gt 1 ]; then 
        # create hostfile, with slots=8 for each node
        scontrol show hostnames $SLURM_JOB_NODELIST | awk '{print $0" slots=8"}' > hostfile.$SLURM_PROCID

        #distributed_args="--no_local_rank ${distributed_args} --hostfile hostfile.$SLURM_PROCID "
        distributed_args=" ${distributed_args} --node_rank=$SLURM_PROCID"

        export NCCL_ASYNC_ERROR_HANDLING=1
    fi

    # if on the main node with PROCID=0, create the model directory and tensorboard directory
    if [ "$SLURM_PROCID" == "0" ]; then
        # if `save_external` is set, symlink it to the `model_dir`
        # and copy the config file to the `model_dir`
        if [ "$external_model_dir" != "" ]; then
            if [ "$external_resume" == false ]; then
                rm -rf $external_model_dir
            fi
            mkdir -p $external_model_dir
            ln -sf $external_model_dir $model_dir
            cp $ds_config $model_dir
        fi

        if [ "$external_tensorboard" != "" ]; then
            mkdir -p $external_tensorboard
            ln -sf $external_tensorboard tensorboard
        else
            mkdir -p tensorboard
        fi
    fi

    ### DEBUG MULTINODE
    if [ "$nodes" -gt 1 ]; then
        $launcher $distributed_args $repo/private_scripts/all_reduce_benchmark.py
    fi
    
    tensorboard_args="--tensorboard-dir tensorboard/ --log-validation-ppl-to-tensorboard"
    micro_batch_size=$(($batch_size / ($gpus * $grad_accum_steps * $nodes)))
    $launcher $distributed_args \
       $repo/Megatron-DeepSpeed/pretrain_gpt.py \
       --tensor-model-parallel-size $tp \
       --no-pipeline-parallel \
       --num-layers $num_layers \
       --hidden-size $hidden_size \
       --ffn-hidden-size $ffn_hidden_size \
       --num-attention-heads $num_attention_heads \
       --num-key-value-heads $num_kv_heads \
       --micro-batch-size $micro_batch_size \
       --global-batch-size $batch_size \
       --checkpoint-activations \
       --seq-length $seq_length \
       --max-position-embeddings $seq_length \
       --train-iters $train_steps \
       --save $model_dir \
       --load $model_dir \
       --data-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $external_tokenizer \
       --split 989,10,1 \
       --distributed-backend nccl \
       --lr $lr \
       --lr-decay-style cosine \
       --min-lr $min_lr \
       --weight-decay $weight_decay \
       --clip-grad $grad_clip \
       --lr-warmup-iters $lr_warmup_steps \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 10 \
       --save-interval $save_interval \
       --eval-interval $eval_interval \
       --eval-iters 10 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --use-flash-attn-v2 \
       --distributed-timeout-minutes 60 \
       --seed $seed \
       $tensorboard_args \
       $ds_args \
       $([ "$multiple_valid_sets" == false ] && echo "--data-path $data_path" || echo "") \
       $([ "$multiple_valid_sets" == true ] && echo "--train-data-path $data_path" || echo "") \
       $([ "$multiple_valid_sets" == true ] && echo "--valid-data-path $valid_data_path" || echo "") \
       $([ "$multiple_valid_sets" == true ] && echo "--multiple-valid-sets" || echo "") \
}


task Eval
    < trained_model=(
        UseExternal:
            false=$model_dir@Train
            true=$external_model_dir
        )
    < eval_set=$test_data@DumpHFDataset
    > eval_results 
    :: .submitter=@ 
    :: .C=$eval_C
    :: .account=$eval_account
    :: .partition=$eval_partition
    :: .time=$eval_time
    :: .cpus=$eval_cpus
    :: .gres=$eval_gres
    :: .mem=$eval_mem
    :: repo=@
    :: gpus=1
    :: batch_size=16
    :: tp=@
    :: pp=@
    :: master_addr=@
    :: master_port=@
    :: eval_metric=@
    :: eval_iteration=@
    :: external_tokenizer=@
    :: model_config=@
{
    # since evals easily clash on the same port due to sharing a machine, find a free one if not free
    master_port=$(python $repo/scripts/find_free_port.py --starting_port $master_port --interval 8)
    echo "Using master_port=$master_port"

    # Read model config and parse to variables
    model_config_f="${repo}/configs/models/${model_config}.yml"
    num_layers=$(yq '.num_layers' $model_config_f)
    num_kv_heads=$(yq '.num_kv_heads' $model_config_f)

    # create simple ds config file
    ds_args="--deepspeed"
    echo -n '{
        "train_batch_size" : '$batch_size',
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": 0
        },
        "bf16": {
            "enabled": true
        }
    }' | jq . > ds_config.json
    ds_args="--zero-stage=0 ${ds_args} --deepspeed_config ds_config.json"
    distributed_args="--num_nodes=1 --num_gpus=$gpus --master_port $master_port"
    launcher="deepspeed"

    echo "Using DeepSpeed for evaluation..."

    micro_batch_size=$(($batch_size / $gpus))
    $launcher $distributed_args \
        $repo/Megatron-DeepSpeed/tasks/main.py \
        --task "WIKITEXT103" \
        --tensor-model-parallel-size $tp \
        --num-layers $num_layers \
        --hidden-size $hidden_size \
        --ffn-hidden-size $ffn_hidden_size \
        --num-attention-heads $num_attention_heads \
        --num-key-value-heads $num_kv_heads \
        --use-rotary-position-embeddings \
        --untie-embeddings-and-output-weights \
        --swiglu \
        --normalization rmsnorm \
        --disable-bias-linear \
        --num-key-value-heads $num_kv_heads \
        --use-flash-attn-v2 \
        --seq-length $seq_length \
        --max-position-embeddings $seq_length \
        --fp16 \
        --valid-data $eval_set \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $external_tokenizer \
        --load $trained_model \
        $( [ "$eval_iteration" != "" ] && echo "--load-iteration $eval_iteration" || echo "") \
        --micro-batch-size $batch_size \
        --log-interval 10 \
        --no-load-optim \
        --no-load-rng \
        --distributed-timeout-minutes 60 \
        $ds_args 
}

task ConvertHF
    < trained_model=(
        UseExternal:
            false=$model_dir@Train
            true=$external_model_dir
        )
    > hf_model_dir=hf_model
    :: .submitter=@
    :: .C=$convert_C
    :: .account=$convert_account
    :: .partition=$convert_partition
    :: .reservation=$convert_reservation
    :: .time=$convert_time 
    :: .cpus=$convert_cpus
    :: .gres=$convert_gres
    :: .mem=$convert_mem
    :: gpus_ids=0
    :: tp=@
    :: pp=@
    :: repo=@
    :: external_tokenizer=@
    :: model_config=@
    :: convert_iteration=@
{
    export CUDA_DEVICE_MAX_CONNECTIONS=1

    export PYTHONPATH=$repo/Megatron-DeepSpeed

    # Read model config and parse to variables
    model_config_f="${repo}/configs/models/${model_config}.yml"
    hidden_size=$(yq '.hidden_size' $model_config_f)
    ffn_hidden_size=$(yq '.ffn_hidden_size' $model_config_f)
    num_layers=$(yq '.num_layers' $model_config_f)
    num_attention_heads=$(yq '.num_attention_heads' $model_config_f)
    seq_length=$(yq '.seq_length' $model_config_f)
    num_kv_heads=$(yq '.num_kv_heads' $model_config_f)

    ds_args="--deepspeed"
    echo -n '{
        "train_batch_size" : 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": 0
        },
        "bf16": {
            "enabled": true
        }
    }' | jq . > ds_config.json
    ds_args="--zero-stage=0 ${ds_args} --deepspeed_config ds_config.json"
    distributed_args="--include localhost:$gpus_ids --master_port 6000"
    
    mkdir -p $hf_model_dir

    deepspeed $distributed_args $repo/scripts/deepspeed_to_hf_llama.py \
        --output-dir $hf_model_dir \
        --tensor-model-parallel-size $tp \
        --pipeline-model-parallel-size $pp \
        --no-pipeline-parallel \
        --num-layers $num_layers  \
        --hidden-size $hidden_size  \
        --max-position-embeddings $seq_length \
        --seq-length $seq_length  \
        --num-attention-heads $num_attention_heads  \
        --ffn-hidden-size $ffn_hidden_size  \
        --no-query-key-layer-scaling \
        --use-rotary-position-embeddings \
        --untie-embeddings-and-output-weights \
        --swiglu \
        --normalization rmsnorm \
        --disable-bias-linear \
        --attention-dropout 0 \
        --hidden-dropout 0 \
        --bf16 \
        --micro-batch-size 1 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $external_tokenizer  \
        --load $trained_model \
        $( [ "$convert_iteration" != "" ] && echo "--load-iteration $convert_iteration" || echo "") \
        --no-save-optim \
        --no-save-rng \
        --no-load-optim \
        --no-load-rng \
        $ds_args
}

plan TrainTokenizer {
    reach TrainTokenizer
}

plan Preprocess {
    reach PreprocessDataset, PreprocessValidDataset via (Dataset: *)
}

plan TrainLLM {
    reach Train via (DataMix: equal)
}

plan EvalLLM {
    reach Eval via (DataMix: equal) * (Dataset: *)
}

#plan ScalingAnalysis {
#    reach Train via (Size: xxs xs s) * (DataMix: enmid enmid_biling_wiki)
#}
 
plan ConvertHF {
    reach ConvertHF via (Size: base) * (ConvertIteration: latest)
}