import deepspeed
import torch
import torch.distributed as D

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig

from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.training import get_model
from megatron.tokenizer import build_tokenizer

from pretrain_gpt import model_provider

from typing import Sequence, Optional


def main():
    initialize_megatron(extra_args_provider=add_save_hf_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    assert args.deepspeed, "This script expects deepspeed to be enabled."

    [ model ] = get_model(model_provider, wrap_with_ddp=False)

    optimizer = None
    opt_param_scheduler = None
    
    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        args=args,
        lr_scheduler=opt_param_scheduler,
        mpu=mpu if args.no_pipeline_parallel else None
    )
    # Deepspeed assumes a list of models
    model = [model]

    args.iteration = load_checkpoint(model, None, None, load_only_weights=True, load_iteration=args.load_iteration)

    output_dir = Path(args.output_dir)

    n_layers = args.num_layers
    n_heads = args.num_attention_heads
    n_heads_kv = args.num_key_value_heads
    n_hidden = args.hidden_size
    intermediate_size = args.ffn_hidden_size
    seq_length = args.seq_length

    assert args.tokenizer_type == "PretrainedFromHF"
    tokenizer = build_tokenizer(args)
    vocab_size = tokenizer.vocab_size

    llama_config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=vocab_size,
        hidden_size=n_hidden,
        intermediate_size=intermediate_size,
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        rms_norm_eps=1e-5,
        num_key_value_heads=n_heads_kv,
        max_position_embeddings=seq_length,
    )

    tp_rank = mpu.get_tensor_model_parallel_rank()
    if tp_rank == 0:
        hf_model = AutoModelForCausalLM.from_config(llama_config)
    else:
        hf_model = None
    
    set_preprocess_state(args, model[0].module, hf_model, vocab_size)
    set_postprocess_state(args, model[0].module, hf_model, vocab_size)
    for layer_idx in range(args.num_layers):
        set_layer_state(args, model[0].module, hf_model, layer_idx)

    if tp_rank == 0:
        print_rank_0(f"> Saving model to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        llama_config.save_pretrained(output_dir)
        # Unwrap the megatron _AutoTokenizer wrapper around
        # the HF AutoTokenizer
        tokenizer.tokenizer.save_pretrained(output_dir)
        hf_model.save_pretrained(output_dir)

        del llama_config
        del tokenizer
        del hf_model

        # Test optimizer model loading
        _ = AutoTokenizer.from_pretrained(output_dir)
        _ = AutoModelForCausalLM.from_pretrained(output_dir)


def set_preprocess_state(args, ds_model, hf_model, vocab_size):
    """Set embedding params.
    
    Embeddings are sharded in dimension 0 (vocab size) so we
    chunk and gather them at dim=0 to obtain the same behaviour.

    (see Megatron-DeepSpeed/tools/checkpoint_{loader,saver}_megatron.py)
    """
    tp_rank = mpu.get_tensor_model_parallel_rank()

    full_size = (vocab_size, args.hidden_size)
    shard = ds_model.language_model.embedding.word_embeddings.weight.data

    full_weight = gather_tp_group(shard, full_size, shard_dim=0, root=0)

    if tp_rank == 0:
        hf_model.model.embed_tokens.weight.data.copy_(full_weight)

def set_postprocess_state(args, ds_model, hf_model, vocab_size):
    """Set final layernorm and output layer params.
    
    Final layernorm is not sharded, so we can just copy it over.
    Output layer is sharded in dimension 0 (vocab size) so we
    chunk and gather it at dim=0 to obtain the same behaviour.
    
    (see Megatron-DeepSpeed/tools/checkpoint_{loader,saver}_megatron.py)
    """   
    tp_rank = mpu.get_tensor_model_parallel_rank()
    
    final_layernorm = ds_model.language_model.encoder.final_layernorm.weight

    output_size = (vocab_size, args.hidden_size)
    shard = ds_model.language_model.output_layer.weight.data
    output = gather_tp_group(shard, output_size, shard_dim=0, root=0)

    if tp_rank == 0:
        hf_model.model.norm.weight.data.copy_(final_layernorm)
        hf_model.lm_head.weight.data.copy_(output)

def set_layer_state(args, ds_model, hf_model, layer_idx):
    """Set layer state.
    
    Layernorms are not sharded, so we can just copy them over.
    """
    tp_rank = mpu.get_tensor_model_parallel_rank()

    ds_layer = ds_model.language_model.encoder.layers[layer_idx]
    input_layernorm = ds_layer.input_layernorm.weight
    post_attention_layernorm = ds_layer.post_attention_layernorm.weight

    if tp_rank == 0:
        hf_layer = hf_model.model.layers[layer_idx]
        hf_layer.input_layernorm.weight.data.copy_(input_layernorm)
        hf_layer.post_attention_layernorm.weight.data.copy_(post_attention_layernorm)
    
    set_attention_state(args, ds_model, hf_model, layer_idx)
    set_mlp_state(args, ds_model, hf_model, layer_idx)

def set_attention_state(args, ds_model, hf_model, layer_idx):
    """Set attention state.

    QKV is sharded in dimension 0 (num kv heads)
    
    Dense is sharded in dimension 1 (hidden size) so we
    chunk and gather it at dim=1 to obtain the same behaviour.
    """

    tp_rank = mpu.get_tensor_model_parallel_rank()

    nh = args.num_attention_heads
    ng = args.num_key_value_heads
    dim = args.kv_channels
    hidden_size = args.hidden_size

    ds_layer = ds_model.language_model.encoder.layers[layer_idx]

    qkv_size = (ng * dim * (nh // ng + 2), hidden_size)
    qkv_shard = ds_layer.self_attention.query_key_value.weight.data
    qkv = gather_tp_group(qkv_shard, qkv_size, shard_dim=0, root=0)

    dense_size = (args.hidden_size, args.hidden_size)
    dense_shard = ds_layer.self_attention.dense.weight.data
    dense = gather_tp_group(dense_shard, dense_size, shard_dim=1, root=0)

    if tp_rank == 0:
        hf_layer = hf_model.model.layers[layer_idx]

        qkv = qkv.reshape((ng,  dim * (nh // ng + 2), hidden_size))

        q_proj = qkv[:, :dim*nh//ng, :]
        k_proj = qkv[:, dim*nh//ng:dim*nh//ng + dim, :]
        v_proj = qkv[:, dim*nh//ng + dim:, :]

        q_proj = q_proj.reshape((ng * dim*nh//ng, -1))
        k_proj = k_proj.reshape((ng * dim, -1))
        v_proj = v_proj.reshape((ng * dim, -1))

        hf_layer.self_attn.q_proj.weight.data.copy_(q_proj)
        hf_layer.self_attn.k_proj.weight.data.copy_(k_proj)
        hf_layer.self_attn.v_proj.weight.data.copy_(v_proj)
        hf_layer.self_attn.o_proj.weight.data.copy_(dense)

def set_mlp_state(args, ds_model, hf_model, layer_idx):
    """Set MLP state.
    
    Dense h to 4h is sharded in dimension 0 (hidden size) so we
    chunk and gather it at dim=0 to obtain the same behaviour.
    We expect this layer to be twice the size as it contains both
    the gate and up projections of the swiglu.

    Dense 4h to h is sharded in dimension 1 (hidden size) so we
    chunk and gather it at dim=1 to obtain the same behaviour.

    (see Megatron-DeepSpeed/tools/checkpoint_{loader,saver}_megatron.py)
    """
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp = args.tensor_model_parallel_size

    ds_layer = ds_model.language_model.encoder.layers[layer_idx]

    # Multiply by 2 as the dense layer contains both the gate and up projections
    dense_h_to_4h_size = (2 * args.ffn_hidden_size, args.hidden_size)
    dense_h_to_4h_shard = ds_layer.mlp.dense_h_to_4h.weight.data
    dense_h_to_4h = gather_tp_group(dense_h_to_4h_shard, dense_h_to_4h_size, shard_dim=0, root=0)

    dense_4h_to_h_size = (args.hidden_size, args.ffn_hidden_size)
    dense_4h_to_h_shard = ds_layer.mlp.dense_4h_to_h.weight.data
    dense_4h_to_h = gather_tp_group(dense_4h_to_h_shard, dense_4h_to_h_size, shard_dim=1, root=0)

    if tp_rank == 0:
        tp_shards = torch.chunk(dense_h_to_4h, tp, dim=0)
        split_shards = [torch.chunk(shard, 2, dim=0) for shard in tp_shards]

        gate_proj = torch.cat([shard[0] for shard in split_shards], dim=0)
        up_proj = torch.cat([shard[1] for shard in split_shards], dim=0)

        hf_layer = hf_model.model.layers[layer_idx]
        hf_layer.mlp.gate_proj.weight.data.copy_(gate_proj)
        hf_layer.mlp.up_proj.weight.data.copy_(up_proj)
        hf_layer.mlp.down_proj.weight.data.copy_(dense_4h_to_h)


def gather_tp_group(
    shard: torch.Tensor, full_size: Sequence[int], shard_dim: int, root: int = 0
) -> Optional[torch.Tensor]:
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if tp_size == 1:
        return shard

    shards = None
    full_size = tuple(full_size)
    assert shard_dim < len(full_size)
    assert full_size[shard_dim] % tp_size == 0

    shard_size = list(full_size)
    shard_size[shard_dim] //= tp_size
    assert shard.size() == torch.Size(shard_size)
    
    if tp_rank == root:
        shards = [torch.empty(shard_size, dtype=shard.dtype, device=shard.device) for _ in range(tp_size)]

    group = mpu.get_tensor_model_parallel_group()
    D.gather(shard, gather_list=shards, dst=root, group=group)

    if tp_rank == root:
        full = torch.cat(shards, dim=shard_dim)
        return full
    
    return None
  

def add_save_hf_args(parser):
    group = parser.add_argument_group(title="Conversion arguments")
    group.add_argument("--output-dir", type=str, required=True)

    return parser

if __name__ == "__main__":
    main()