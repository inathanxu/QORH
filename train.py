import os
import argparse
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import logging
from load_data import DataLoader
from base_model import BaseModel
from models import RED_GNN_induc
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset training and inference for GraphOracle"
    )

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'zero-shot', 'finetune'],
                        help="Running mode: train, zero-shot, or finetune")

    parser.add_argument('--datasets', nargs='+', required=True,
                        help="List of dataset paths; either omit the _ind "
                             "suffix or pass the full directory in "
                             "fb237_v1_train_ind form")

    parser.add_argument('--model_path', type=str,
                        help="Path to the pretrained model (required for "
                             "zero-shot and finetune modes)")
    parser.add_argument('--num_layers', type=int, default=3,
                        help="Number of relation encoder layers")
    parser.add_argument('--max_rules_per_query', type=int, default=4,
                        help="Maximum number of rule hyperedges selected "
                             "per query relation (default 4)")
    parser.add_argument('--max_rule_length', type=int, default=2,
                        help="Maximum length of the path rule body "
                             "(default 2)")
    parser.add_argument('--min_rule_support', type=int, default=2,
                        help="Minimum number of supporting entity pairs "
                             "for a candidate rule (default 2)")
    parser.add_argument('--rule_mlp_hidden', type=int, default=16,
                        help="Hidden dimension of the rule joint-selection "
                             "score MLP (default 16)")
    parser.add_argument('--rule_attention_eta', type=float, default=1.0,
                        help="Prior strength of rule scores in hyperedge "
                             "attention (default 1.0)")
    parser.add_argument('--rule_max_neighbors', type=int, default=64,
                        help="Maximum outgoing edges used per entity during "
                             "rule mining; 0 means unlimited")
    parser.add_argument('--rule_max_paths_per_entity', type=int, default=1024,
                        help="Maximum paths enumerated per start entity and "
                             "length during rule mining; 0 means unlimited")

    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (for reproducibility)")
    parser.add_argument('--num_gpus', type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID used in single-GPU mode")
    parser.add_argument('--gpu_ids', nargs='+', type=int,
                        help="Specific GPU IDs to use for multi-GPU training")
    parser.add_argument('--max_epochs', type=int, default=200,
                        help="Maximum number of training epochs")
    parser.add_argument('--early_stopping', type=int, default=10,
                        help="Early stopping patience: stop training after "
                             "this many epochs without improvement")
    parser.add_argument('--finetune_epochs', type=int, default=50,
                        help="Number of epochs for finetune mode")

    parser.add_argument('--save_model', action='store_true',
                        help="Save the model after training on each dataset")
    parser.add_argument('--model_dir', type=str, default='saved_models',
                        help="Directory for saving models")
    parser.add_argument('--keep_history', action='store_true',
                        help="Compatibility with old scripts: additionally "
                             "save model_<dataset>.pt; by default "
                             "<dataset>.pt and latest_model.pt are always "
                             "saved")

    parser.add_argument('--rel_single', action='store_true',
                        help="Use single-direction relations (instead of "
                             "bidirectional)")
    parser.add_argument('--rel_reverse_only', action='store_true',
                        help="Use reverse relations only")
    parser.add_argument('--fact_ratio', type=float, default=0.9,
                        help="Fraction of fact data in the total training "
                             "data (default 0.9)")
    parser.add_argument('--remove_1hop_edges', action='store_true',
                        help="Remove 1-hop edges to prevent information "
                             "leakage")

    parser.add_argument('--log_file', type=str, default='red_gnn.log',
                        help="Log file path")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="Number of attention heads")

    args = parser.parse_args()

    if args.num_gpus > 1 and args.gpu_ids and len(args.gpu_ids) != args.num_gpus:
        parser.error(
            f"When specifying gpu_ids, provide exactly {args.num_gpus} GPU IDs"
        )
    if args.early_stopping < 1:
        parser.error("--early_stopping must be an integer >= 1")
    if args.max_rules_per_query < 1:
        parser.error("--max_rules_per_query must be >= 1")
    if args.max_rule_length < 1:
        parser.error("--max_rule_length must be >= 1")
    if args.min_rule_support < 1:
        parser.error("--min_rule_support must be >= 1")
    if args.rule_mlp_hidden < 1:
        parser.error("--rule_mlp_hidden must be >= 1")
    if args.rule_attention_eta < 0:
        parser.error("--rule_attention_eta must not be negative")
    if args.rule_max_neighbors < 0:
        parser.error("--rule_max_neighbors must not be negative")
    if args.rule_max_paths_per_entity < 0:
        parser.error("--rule_max_paths_per_entity must not be negative")
    if args.mode == 'train' and args.max_epochs < 1:
        parser.error("--max_epochs must be >= 1 in train mode")
    if args.mode == 'finetune' and args.finetune_epochs < 1:
        parser.error("--finetune_epochs must be >= 1 in finetune mode")

    setup_logging(args)

    if args.mode == 'train':
        multi_dataset_train(args)
    elif args.mode == 'zero-shot':
        if not args.model_path:
            logging.error("zero-shot mode requires --model_path")
            return
        zero_shot_inference(args)
    elif args.mode == 'finetune':
        if not args.model_path:
            logging.error("finetune mode requires --model_path")
            return
        finetune_inference(args)


def setup_logging(args):
    logging.basicConfig(
        filename=args.log_file,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


def setup(rank, world_size, gpu_ids):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if gpu_ids and rank < len(gpu_ids):
        gpu_id = gpu_ids[rank]
        torch.cuda.set_device(gpu_id)
    else:
        torch.cuda.set_device(rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_dataset_label(dataset_path):
    return os.path.basename(os.path.normpath(dataset_path))


def get_hyperparam_dataset_name(dataset_label):
    for suffix in ('_train_ind', '_ind', '_train'):
        if dataset_label.endswith(suffix):
            return dataset_label[:-len(suffix)]
    return dataset_label


def get_dataset_model_path(model_dir, dataset_label):
    return os.path.join(model_dir, f"{dataset_label}.pt")


def build_data_loader(dataset_path, args):
    return DataLoader(
        dataset_path,
        getattr(args, 'rel_single', False),
        getattr(args, 'rel_reverse_only', False),
        getattr(args, 'fact_ratio', 0.9),
        getattr(args, 'remove_1hop_edges', False),
        max_rule_length=getattr(args, 'max_rule_length', 2),
        min_rule_support=getattr(args, 'min_rule_support', 2),
        rule_max_neighbors=getattr(args, 'rule_max_neighbors', 64),
        rule_max_paths_per_entity=getattr(
            args, 'rule_max_paths_per_entity', 1024
        ),
    )


def apply_rule_model_options(opts, args):
    opts.max_rules_per_query = getattr(args, 'max_rules_per_query', 4)
    opts.max_rule_length = getattr(args, 'max_rule_length', 2)
    opts.min_rule_support = getattr(args, 'min_rule_support', 2)
    opts.rule_mlp_hidden = getattr(args, 'rule_mlp_hidden', 16)
    opts.rule_attention_eta = getattr(args, 'rule_attention_eta', 1.0)
    opts.rule_max_neighbors = getattr(args, 'rule_max_neighbors', 64)
    opts.rule_max_paths_per_entity = getattr(
        args, 'rule_max_paths_per_entity', 1024
    )
    return opts


def get_model_architecture(model):
    return {
        'hidden_dim': model.hidden_dim,
        'attn_dim': model.attn_dim,
        'n_layer': model.n_layer,
        'relation_encoder_layers': model.relation_encoder.num_layers,
        'relation_encoder_heads': model.relation_encoder.num_heads,
        'max_rule_length': model.max_rule_length,
        'max_rules_per_query': model.max_rules_per_query,
        'rule_mlp_hidden': model.rule_mlp_hidden,
        'rule_attention_eta': model.rule_attention_eta,
        'min_rule_support': model.min_rule_support,
        'rule_max_neighbors': model.rule_max_neighbors,
        'rule_max_paths_per_entity': model.rule_max_paths_per_entity,
    }


def checkpoint_matches_paired_architecture(checkpoint, target_model,
                                           target_config_name,
                                           target_dataset_name):
    source_config_name = checkpoint.get('config_dataset')
    source_architecture = checkpoint.get('model_architecture')
    if (source_config_name != target_config_name or
            source_architecture is None):
        return True

    target_architecture = get_model_architecture(target_model)
    differences = {
        key: (source_architecture.get(key), target_value)
        for key, target_value in target_architecture.items()
        if source_architecture.get(key) != target_value
    }
    if differences:
        logging.error(
            f"checkpoint model architecture is inconsistent with "
            f"{target_dataset_name}: {differences}. Please ensure training "
            "and testing use the same relation encoding and rule hypergraph "
            "parameters."
        )
        return False
    return True


def save_model(model, optimizer, scheduler, epoch, dataset_name, args,
               config_dataset_name=None, validation_mrr=None):
    os.makedirs(args.model_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'last_dataset': dataset_name,
        'validation_mrr': validation_mrr,
        'config_dataset': config_dataset_name or dataset_name,
        'model_architecture': get_model_architecture(model),
    }

    dataset_save_path = get_dataset_model_path(args.model_dir, dataset_name)
    torch.save(checkpoint, dataset_save_path)

    latest_save_path = os.path.join(args.model_dir, "latest_model.pt")
    torch.save(checkpoint, latest_save_path)

    if getattr(args, 'keep_history', False):
        legacy_path = os.path.join(args.model_dir, f"model_{dataset_name}.pt")
        torch.save(checkpoint, legacy_path)
        logging.info(
            f"Model saved to {dataset_save_path} and {latest_save_path}; "
            f"compatibility copy saved to {legacy_path}"
        )
    else:
        logging.info(
            f"Model saved to {dataset_save_path}; latest model alias is "
            f"{latest_save_path}"
        )

    return dataset_save_path


def load_model(model, optimizer, scheduler, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        logging.info(f"Loaded model from {checkpoint_path}")
        return start_epoch + 1
    return 0


def transfer_compatible_params(target_model, source_state_dict, rank=0):
    target_state_dict = target_model.state_dict()

    total_params = len(target_state_dict)
    transferred = 0

    compatible_params = {}
    for name, param in target_state_dict.items():
        if (name in source_state_dict and
                param.shape == source_state_dict[name].shape):
            compatible_params[name] = source_state_dict[name]
            transferred += 1

    target_state_dict.update(compatible_params)
    target_model.load_state_dict(target_state_dict)

    if rank == 0:
        logging.info(
            f"Transferred {transferred}/{total_params} parameters from the "
            "previous model"
        )
        if transferred < total_params:
            logging.info(
                "Model architectures differ — only compatible parameters "
                "were transferred"
            )

    return transferred > 0


def train_model(rank, world_size, args, dataset_path, previous_model_state=None):
    if world_size > 1:
        setup(rank, world_size, args.gpu_ids)
    elif torch.cuda.is_available():
        torch.cuda.set_device(rank)

    is_main_process = world_size == 1 or rank == 0

    if world_size > 1 and args.gpu_ids and rank < len(args.gpu_ids):
        device = torch.device(f"cuda:{args.gpu_ids[rank]}")
    else:
        device = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        )

    results_dir = 'results'
    if is_main_process and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    loader = build_data_loader(dataset_path, args)

    dataset_name = get_dataset_label(loader.ind_dir)
    config_dataset_name = get_hyperparam_dataset_name(dataset_name)

    opts = get_dataset_hyperparams(config_dataset_name)
    opts.n_ent = loader.n_ent
    opts.n_ent_ind = getattr(loader, 'n_ent_ind', loader.n_ent)
    opts.n_rel = loader.n_rel
    opts.n_rel_ind = loader.n_rel_ind
    opts.num_layers = args.num_layers
    apply_rule_model_options(opts, args)
    opts.perf_file = os.path.join(results_dir, f"{dataset_name}_perf.txt")

    base_model = BaseModel(opts, loader)
    base_model.model.to(device)

    if previous_model_state is not None:
        if is_main_process:
            logging.info(
                "Attempting to transfer compatible parameters from the "
                "previous model..."
            )
        transfer_compatible_params(
            base_model.model, previous_model_state,
            rank=0 if is_main_process else rank
        )

    if world_size > 1:
        model = DDP(
            base_model.model,
            device_ids=[args.gpu_ids[rank]]
            if args.gpu_ids and rank < len(args.gpu_ids) else None
        )
    else:
        model = base_model.model

    if is_main_process:
        config_str = (
            f'Dataset: {dataset_name}, Config: {config_dataset_name}, '
            f'LR: {opts.lr:.6f}, '
            f'Decay: {opts.decay_rate:.6f}, Lambda: {opts.lamb:.6f}, '
            f'Hidden: {opts.hidden_dim}, Attn: {opts.attn_dim}, '
            f'Layers: {opts.n_layer}, Batch: {opts.n_batch}, '
            f'Dropout: {opts.dropout:.4f}, Act: {opts.act}, '
            f'Fact ratio: {args.fact_ratio:.2f}, '
            f'Rules/query: {opts.max_rules_per_query}, '
            f'Rule length: {opts.max_rule_length}, '
            f'Rule support: {opts.min_rule_support}, '
            f'Rule MLP: {opts.rule_mlp_hidden}'
        )
        logging.info(config_str)

        if world_size > 1:
            gpu_str = (
                f"Using GPUs: {args.gpu_ids if args.gpu_ids else list(range(world_size))}"
            )
            logging.info(gpu_str)

    if is_main_process:
        epoch_pbar = tqdm(
            total=args.max_epochs,
            desc=f"Training on {dataset_name}",
            position=0, leave=True, dynamic_ncols=True
        )

    best_mrr = float('-inf')
    best_epoch = 0
    best_str = ""
    best_model_state_in_memory = None
    early_stop_counter = 0

    for epoch in range(args.max_epochs):
        v_mrr, _, out_str = base_model.train_batch(
            model if world_size > 1 else None
        )

        loader.shuffle_train()

        if is_main_process:
            with open(opts.perf_file, 'a+') as f:
                f.write(f"Epoch {epoch}: {out_str}")

        if v_mrr > best_mrr:
            best_mrr = v_mrr
            best_str = out_str
            best_epoch = epoch
            best_model_state_in_memory = copy.deepcopy(
                base_model.model.state_dict()
            )
            early_stop_counter = 0

            if is_main_process and args.save_model:
                save_model(
                    base_model.model, base_model.optimizer,
                    base_model.scheduler, epoch, dataset_name, args,
                    config_dataset_name=config_dataset_name,
                    validation_mrr=float(v_mrr)
                )

            if is_main_process:
                logging.info(
                    f"Epoch {epoch}: new best performance — {best_str.strip()}"
                )
        else:
            early_stop_counter += 1
            if is_main_process:
                logging.info(
                    f"Epoch {epoch}: no improvement, "
                    f"counter: {early_stop_counter}/{args.early_stopping}"
                )

        if is_main_process:
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                'val_MRR': f"{v_mrr:.4f}",
                'best_val_MRR': f"{best_mrr:.4f}",
                'counter': f"{early_stop_counter}/{args.early_stopping}"
            })

        if world_size > 1:
            early_stop_tensor = torch.tensor(
                [early_stop_counter], device=device
            )
            dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.MAX)
            early_stop_counter = early_stop_tensor.item()

        if early_stop_counter >= args.early_stopping:
            if is_main_process:
                logging.info(
                    f"Early stopping triggered after {epoch} epochs"
                )
            break

    if is_main_process:
        epoch_pbar.close()
        logging.info(
            f"Best checkpoint selected by valid.txt on {dataset_name} "
            f"(epoch {best_epoch}): {best_str.strip()}. "
            "Test queries and metrics were not used for training or "
            "model selection; load this checkpoint in test/zero-shot "
            "mode for final testing."
        )

    best_model_state = None
    if args.save_model:
        checkpoint_path = get_dataset_model_path(args.model_dir, dataset_name)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            best_model_state = checkpoint['model_state_dict']
    else:
        best_model_state = best_model_state_in_memory

    if world_size > 1:
        cleanup()

    return best_model_state


def multi_dataset_train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.save_model and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    logging.info(
        f"Starting sequential training on {len(args.datasets)} datasets"
    )

    fixed_model_path = os.path.join(args.model_dir, "latest_model.pt")

    dataset_pbar = tqdm(
        total=len(args.datasets),
        desc="Datasets Progress",
        position=0, leave=True, dynamic_ncols=True
    )

    previous_model_state = None
    for dataset_idx, dataset_path in enumerate(args.datasets):
        logging.info(
            f"Training dataset {dataset_idx+1}/{len(args.datasets)}: "
            f"{dataset_path}"
        )

        if dataset_idx > 0 and os.path.exists(fixed_model_path) and args.save_model:
            device = torch.device(
                f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
            )
            checkpoint = torch.load(
                fixed_model_path, map_location=device, weights_only=True
            )
            previous_model_state = checkpoint['model_state_dict']
            last_dataset = checkpoint.get('last_dataset', 'unknown')
            logging.info(
                f"Loaded the model trained on {last_dataset} to continue "
                "training"
            )

        if args.num_gpus > 1:
            mp.spawn(
                train_model,
                args=(args.num_gpus, args, dataset_path, previous_model_state),
                nprocs=args.num_gpus,
                join=True
            )

            if os.path.exists(fixed_model_path) and args.save_model:
                device = torch.device(
                    f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
                )
                checkpoint = torch.load(
                    fixed_model_path, map_location=device, weights_only=True
                )
                previous_model_state = checkpoint['model_state_dict']
        else:
            previous_model_state = train_model(
                args.gpu, 1, args, dataset_path, previous_model_state
            )

        dataset_pbar.update(1)
        dataset_name = (
            dataset_path.split('/')[-1]
            if dataset_path.split('/')[-1]
            else dataset_path.split('/')[-2]
        )
        dataset_pbar.set_description(
            f"Completed {dataset_idx+1}/{len(args.datasets)}: {dataset_name}"
        )

    dataset_pbar.close()
    logging.info("Sequential training completed!")


def zero_shot_inference(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    if not os.path.exists(args.model_path):
        logging.error(f"Model file does not exist: {args.model_path}")
        return

    logging.info(f"Running zero-shot inference with model: {args.model_path}")

    orig_checkpoint = torch.load(
        args.model_path, map_location=device, weights_only=True
    )
    orig_state_dict = orig_checkpoint['model_state_dict']
    selected_epoch = orig_checkpoint.get('epoch', 'unknown')
    selected_valid_mrr = orig_checkpoint.get('validation_mrr')
    if selected_valid_mrr is not None:
        logging.info(
            f"Loaded the checkpoint selected by valid.txt "
            f"(epoch={selected_epoch}, validation MRR="
            f"{float(selected_valid_mrr):.4f}); this mode only evaluates "
            "test.txt."
        )
    else:
        logging.info(
            f"Loaded checkpoint (epoch={selected_epoch}); this "
            "legacy-format file has no recorded validation MRR, and this "
            "mode still only evaluates test.txt."
        )

    dataset_pbar = tqdm(
        total=len(args.datasets),
        desc="Zero-shot Inference",
        position=0, leave=True, dynamic_ncols=True
    )

    for dataset_idx, dataset_path in enumerate(args.datasets):
        loader = build_data_loader(dataset_path, args)

        dataset_name = get_dataset_label(loader.ind_dir)
        config_dataset_name = get_hyperparam_dataset_name(dataset_name)
        logging.info(
            f"Running zero-shot inference on dataset: {dataset_name} "
            f"(config: {config_dataset_name})"
        )
        dataset_pbar.set_description(f"Zero-shot on {dataset_name}")

        opts = get_dataset_hyperparams(config_dataset_name)
        opts.n_ent = loader.n_ent
        opts.n_ent_ind = getattr(loader, 'n_ent_ind', loader.n_ent)
        opts.n_rel = loader.n_rel
        opts.n_rel_ind = loader.n_rel_ind
        opts.num_layers = args.num_layers
        apply_rule_model_options(opts, args)

        base_model = BaseModel(opts, loader)
        base_model.model.to(device)

        if not checkpoint_matches_paired_architecture(
                orig_checkpoint, base_model.model, config_dataset_name,
                dataset_name):
            dataset_pbar.update(1)
            continue

        success = transfer_compatible_params(
            base_model.model, orig_state_dict
        )

        if not success:
            logging.warning(
                f"Model architectures differ too much to run zero-shot "
                f"inference on {dataset_name}"
            )
            dataset_pbar.update(1)
            continue

        _, t_mrr, out_str = base_model.evaluate(split='test')
        logging.info(
            f"Final test results on {dataset_name}: {out_str.strip()}"
        )
        dataset_pbar.update(1)

    dataset_pbar.close()


def finetune_inference(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    if not os.path.exists(args.model_path):
        logging.error(f"Model file does not exist: {args.model_path}")
        return

    logging.info(f"Fine-tuning model from: {args.model_path}")

    checkpoint = torch.load(
        args.model_path, map_location=device, weights_only=True
    )
    model_state = checkpoint['model_state_dict']

    dataset_pbar = tqdm(
        total=len(args.datasets),
        desc="Fine-tuning Datasets",
        position=0, leave=True, dynamic_ncols=True
    )

    for dataset_idx, dataset_path in enumerate(args.datasets):
        loader = build_data_loader(dataset_path, args)

        dataset_name = get_dataset_label(loader.ind_dir)
        config_dataset_name = get_hyperparam_dataset_name(dataset_name)
        logging.info(
            f"Fine-tuning on dataset: {dataset_name} "
            f"(config: {config_dataset_name})"
        )
        dataset_pbar.set_description(f"Fine-tuning on {dataset_name}")

        opts = get_dataset_hyperparams(config_dataset_name)
        opts.n_ent = loader.n_ent
        opts.n_ent_ind = getattr(loader, 'n_ent_ind', loader.n_ent)
        opts.n_rel = loader.n_rel
        opts.n_rel_ind = loader.n_rel_ind
        opts.num_layers = args.num_layers
        apply_rule_model_options(opts, args)

        base_model = BaseModel(opts, loader)
        base_model.model.to(device)

        if not checkpoint_matches_paired_architecture(
                checkpoint, base_model.model, config_dataset_name,
                dataset_name):
            dataset_pbar.update(1)
            continue

        success = transfer_compatible_params(
            base_model.model, model_state
        )

        if not success:
            logging.warning(
                f"Model architectures differ too much to fine-tune on "
                f"{dataset_name}"
            )
            dataset_pbar.update(1)
            continue

        epoch_pbar = tqdm(
            total=args.finetune_epochs,
            desc=f"Fine-tuning epochs for {dataset_name}",
            position=1, leave=False, dynamic_ncols=True
        )

        logging.info(f"Fine-tuning for {args.finetune_epochs} epochs...")
        best_mrr = float('-inf')
        best_str = ""
        best_epoch = 0
        best_finetuned_state = None

        for epoch in range(args.finetune_epochs):
            v_mrr, _, out_str = base_model.train_batch()
            loader.shuffle_train()

            if v_mrr > best_mrr:
                best_mrr = v_mrr
                best_str = out_str
                best_epoch = epoch
                best_finetuned_state = copy.deepcopy(
                    base_model.model.state_dict()
                )
                epoch_pbar.set_postfix({
                    'val_MRR': f"{v_mrr:.4f}",
                    'best_val_MRR': f"{best_mrr:.4f}"
                })
                logging.info(f"Epoch {epoch}: {best_str.strip()}")

            epoch_pbar.update(1)

        epoch_pbar.close()

        if best_finetuned_state is None:
            raise RuntimeError(
                f"No usable valid checkpoint was produced during "
                f"fine-tuning on {dataset_name}"
            )
        base_model.model.load_state_dict(best_finetuned_state)
        logging.info(
            f"Fine-tuning selected epoch {best_epoch} via valid.txt "
            f"(MRR={best_mrr:.4f}); now evaluating only test.txt."
        )
        _, t_mrr, out_str = base_model.evaluate(split='test')
        logging.info(
            f"Fine-tuning results on {dataset_name}: {out_str.strip()}"
        )
        dataset_pbar.update(1)

    dataset_pbar.close()


if __name__ == "__main__":
    main()
