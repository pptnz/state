import argparse as ap


def add_arguments_predict(parser: ap.ArgumentParser):
    """
    CLI for evaluation using cell-eval metrics.
    """

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--toml",
        type=str,
        default=None,
        help="Optional path to a TOML data config to use instead of the training config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename. Default is 'last.ckpt'. Relative to the output directory.",
    )

    parser.add_argument(
        "--test-time-finetune",
        type=int,
        default=0,
        help="If >0, run test-time fine-tuning for the specified number of epochs on only control cells.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "minimal", "de", "anndata"],
        help="run all metrics, minimal, only de metrics, or only output adatas",
    )

    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="If set, only run prediction without evaluation metrics.",
    )

    parser.add_argument(
        "--shared-only",
        action="store_true",
        help=("If set, restrict predictions/evaluation to perturbations shared between train and test (train ∩ test)."),
    )

    parser.add_argument(
        "--eval-train-data",
        action="store_true",
        help="If set, evaluate the model on the training data rather than on the test data.",
    )

    parser.add_argument(
        "--split-by-context",
        action="store_true",
        help=(
            "If set, split predictions by context and write per-context adata_pred_{context}.h5ad and "
            "adata_real_{context}.h5ad files using a bounded-memory two-pass strategy. "
            "Uses dataset_name + cell_type when available, otherwise falls back to data.kwargs.cell_type_key."
        ),
    )


def run_tx_predict(args: ap.ArgumentParser):
    import logging
    import os
    import sys
    import tempfile
    from collections import OrderedDict

    import anndata
    import lightning.pytorch as pl
    import numpy as np
    import pandas as pd
    from scipy import sparse as sp
    import torch
    import yaml

    # Cell-eval for metrics computation
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype
    from cell_load.data_modules import PerturbationDataModule
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.multiprocessing.set_sharing_strategy("file_system")

    def run_test_time_finetune(model, dataloader, ft_epochs, control_pert, device):
        """
        Perform test-time fine-tuning on only control cells.
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        logger.info(f"Starting test-time fine-tuning for {ft_epochs} epoch(s) on control cells only.")
        for epoch in range(ft_epochs):
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"Finetune epoch {epoch + 1}/{ft_epochs}", leave=True)
            for batch in pbar:
                # Check if this batch contains control cells
                first_pert = (
                    batch["pert_name"][0] if isinstance(batch["pert_name"], list) else batch["pert_name"][0].item()
                )
                if first_pert != control_pert:
                    continue

                # Move batch data to device
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                optimizer.zero_grad()
                loss = model.training_step(batch, batch_idx=0, padded=False)
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            logger.info(f"Finetune epoch {epoch + 1}/{ft_epochs}, mean loss: {mean_loss}")
        model.eval()

    def load_config(cfg_path: str) -> dict:
        """Load config from the YAML file that was dumped during training."""
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    def clip_anndata_values(adata: anndata.AnnData, max_value: float, min_value: float = 0.0) -> None:
        """Clip adata.X values in-place to keep cell-eval scale checks happy."""
        if sp.issparse(adata.X):
            # Clip only the stored data to keep sparsity intact.
            if adata.X.data.size:
                np.clip(adata.X.data, min_value, max_value, out=adata.X.data)
                if hasattr(adata.X, "eliminate_zeros"):
                    adata.X.eliminate_zeros()
        else:
            np.clip(adata.X, min_value, max_value, out=adata.X)

    def infer_batch_size(batch: dict) -> int:
        for key in ("pert_cell_emb", "pert_cell_counts", "ctrl_cell_emb", "batch", "pert_name", "cell_type"):
            if key not in batch:
                continue
            value = batch[key]
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
            if isinstance(value, np.ndarray):
                return int(value.shape[0]) if value.ndim > 0 else 1
            if isinstance(value, (list, tuple)):
                return len(value)
        raise ValueError("Unable to infer batch size from batch contents.")

    def normalize_batch_labels(values, batch_size: int):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if isinstance(values, np.ndarray):
            if values.ndim == 2:
                if values.shape[0] != batch_size:
                    return None
                if values.shape[1] == 1:
                    flat = values.reshape(batch_size)
                    return [str(x) for x in flat.tolist()]
                indices = values.argmax(axis=1)
                return [str(int(x)) for x in indices.tolist()]
            if values.ndim == 1:
                if values.shape[0] != batch_size:
                    return None
                return [str(x) for x in values.tolist()]
            if values.ndim == 0:
                return [str(values.item())] * batch_size
            return None
        if isinstance(values, (list, tuple)):
            if len(values) != batch_size:
                return None
            normalized = []
            for item in values:
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                if isinstance(item, np.ndarray):
                    if item.ndim == 0:
                        normalized.append(str(item.item()))
                        continue
                    if item.ndim == 1:
                        if item.size == 1:
                            normalized.append(str(item.item()))
                        elif np.count_nonzero(item) == 1:
                            normalized.append(str(int(item.argmax())))
                        else:
                            normalized.append(str(item.tolist()))
                        continue
                normalized.append(str(item))
            return normalized
        return [str(values)] * batch_size

    def get_batch_labels(candidates, batch_size: int):
        batch_labels = None
        for candidate in candidates:
            batch_labels = normalize_batch_labels(candidate, batch_size)
            if batch_labels is not None:
                break
        if batch_labels is None:
            batch_labels = ["None"] * batch_size
        return batch_labels

    def resolve_context_labels(batch: dict, batch_size: int, fallback):
        dataset_labels = None
        for key in ("dataset_name", "dataset"):
            if key in batch and batch.get(key) is not None:
                dataset_labels = normalize_batch_labels(batch.get(key), batch_size)
                if dataset_labels is not None:
                    break
        context_labels = normalize_batch_labels(fallback, batch_size) if fallback is not None else None

        if dataset_labels is not None and context_labels is not None:
            combined = [f"{ds}.{ct}" for ds, ct in zip(dataset_labels, context_labels)]
            return combined, "dataset_name+cell_type"
        if context_labels is not None:
            return context_labels, "cell_type"
        return None, None

    def ensure_list(values, batch_size: int):
        if isinstance(values, list):
            return values
        if isinstance(values, tuple):
            return list(values)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if isinstance(values, np.ndarray):
            if values.ndim == 0:
                return [values.item()] * batch_size
            return values.tolist()
        return [values] * batch_size


    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    if args.toml:
        data_section = cfg.get("data")
        if data_section is None or "kwargs" not in data_section:
            raise KeyError("The loaded config does not contain data.kwargs, unable to override toml_config_path.")
        cfg["data"]["kwargs"]["toml_config_path"] = args.toml
        logger.info("Overriding data.kwargs.toml_config_path to %s", args.toml)

    # 2. Find run output directory & load data module
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])
    data_module_path = os.path.join(run_output_dir, "data_module.torch")
    if not os.path.exists(data_module_path):
        raise FileNotFoundError(f"Could not find data module at {data_module_path}?")
    data_module = PerturbationDataModule.load_state(data_module_path)
    if args.toml:
        if not os.path.exists(args.toml):
            raise FileNotFoundError(f"Could not find TOML config file at {args.toml}")
        from cell_load.config import ExperimentConfig

        logger.info("Reloading data module configuration from %s", args.toml)
        data_module.toml_config_path = args.toml
        data_module.config = ExperimentConfig.from_toml(args.toml)
        data_module.config.validate()
        data_module.train_datasets = []
        data_module.val_datasets = []
        data_module.test_datasets = []
        data_module._setup_global_maps()
    data_module.setup(stage="test")
    logger.info("Loaded data module from %s", data_module_path)

    # Seed everything
    pl.seed_everything(cfg["training"]["train_seed"])

    # 3. Load the trained model
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    # Determine model class and load
    model_class_name = cfg["model"]["name"]
    model_kwargs = cfg["model"]["kwargs"]

    # Import the correct model class
    if model_class_name.lower() == "embedsum":
        from ...tx.models.embed_sum import EmbedSumPerturbationModel

        ModelClass = EmbedSumPerturbationModel
    elif model_class_name.lower() == "old_neuralot":
        from ...tx.models.old_neural_ot import OldNeuralOTPerturbationModel

        ModelClass = OldNeuralOTPerturbationModel
    elif model_class_name.lower() in ["neuralot", "pertsets", "state"]:
        from ...tx.models.state_transition import StateTransitionPerturbationModel

        ModelClass = StateTransitionPerturbationModel

    elif model_class_name.lower() in ["globalsimplesum", "perturb_mean"]:
        from ...tx.models.perturb_mean import PerturbMeanPerturbationModel

        ModelClass = PerturbMeanPerturbationModel
    elif model_class_name.lower() in ["celltypemean", "context_mean"]:
        from ...tx.models.context_mean import ContextMeanPerturbationModel

        ModelClass = ContextMeanPerturbationModel
    elif model_class_name.lower() == "decoder_only":
        from ...tx.models.decoder_only import DecoderOnlyPerturbationModel

        ModelClass = DecoderOnlyPerturbationModel
    elif model_class_name.lower() == "pseudobulk":
        from ...tx.models.pseudobulk import PseudobulkPerturbationModel

        ModelClass = PseudobulkPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims = data_module.get_var_dims()
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        **model_kwargs,
    }

    model = ModelClass.load_from_checkpoint(checkpoint_path, weights_only=False, **model_init_kwargs)
    model.eval()
    logger.info("Model loaded successfully.")

    # 4. Test-time fine-tuning if requested
    data_module.batch_size = 1
    if args.test_time_finetune > 0:
        control_pert = data_module.get_control_pert()
        if args.eval_train_data:
            test_loader = data_module.train_dataloader(test=True)
        else:
            test_loader = data_module.test_dataloader()

        run_test_time_finetune(
            model, test_loader, args.test_time_finetune, control_pert, device=next(model.parameters()).device
        )
        logger.info("Test-time fine-tuning complete.")

    # 5. Run inference on test set
    data_module.setup(stage="test")
    if args.eval_train_data:
        test_loader = data_module.train_dataloader(test=True)
    else:
        test_loader = data_module.test_dataloader()

    if test_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    num_cells = test_loader.batch_sampler.tot_num
    output_dim = var_dims["output_dim"]
    gene_dim = var_dims["gene_dim"]
    hvg_dim = var_dims["hvg_dim"]

    logger.info("Generating predictions on test set using manual loop...")
    device = next(model.parameters()).device

    cfg_batch_col = cfg.get("data", {}).get("kwargs", {}).get("batch_col", None)
    batch_obs_key = cfg_batch_col or data_module.batch_col

    store_raw_expression = (
        data_module.embed_key is not None
        and data_module.embed_key != "X_hvg"
        and cfg["data"]["kwargs"]["output_space"] == "gene"
    ) or (data_module.embed_key is not None and cfg["data"]["kwargs"]["output_space"] == "all")

    if args.split_by_context:
        context_key = cfg.get("data", {}).get("kwargs", {}).get("cell_type_key", None) or data_module.cell_type_key
        logger.info("Split-by-context enabled; counting cells per context.")

        context_counts: OrderedDict[str, int] = OrderedDict()
        context_order: list[str] = []
        context_mode = None
        for batch in tqdm(test_loader, desc="Counting contexts", unit="batch"):
            batch_size = infer_batch_size(batch)
            context_labels, detected_mode = resolve_context_labels(batch, batch_size, batch.get("cell_type"))
            if context_labels is None:
                raise ValueError(
                    "split-by-context requires dataset_name + cell_type (preferred) or cell_type alone. "
                    f"Check data.kwargs.cell_type_key (current: {context_key})."
                )
            if context_mode is None:
                context_mode = detected_mode
                logger.info("Split-by-context using '%s'.", context_mode)
            elif detected_mode != context_mode:
                raise ValueError(
                    f"Inconsistent context source during counting: saw '{detected_mode}' after '{context_mode}'."
                )
            for label in context_labels:
                if label not in context_counts:
                    context_counts[label] = 0
                    context_order.append(label)
                context_counts[label] += 1

        if len(context_counts) == 0:
            logger.warning("No contexts found in test dataloader. Exiting.")
            sys.exit(0)

        context_slugs = {ctx: str(ctx).replace("/", "-") for ctx in context_order}

        if args.eval_train_data:
            results_dir = os.path.join(args.output_dir, "eval_train_" + os.path.basename(args.checkpoint))
        else:
            results_dir = os.path.join(args.output_dir, "eval_" + os.path.basename(args.checkpoint))
        os.makedirs(results_dir, exist_ok=True)

        tmp_dir = tempfile.TemporaryDirectory(prefix="predict_split_")
        context_store = {}
        for ctx in context_order:
            ctx_slug = context_slugs[ctx]
            ctx_count = context_counts[ctx]
            preds_path = os.path.join(tmp_dir.name, f"{ctx_slug}_preds.mmap")
            reals_path = os.path.join(tmp_dir.name, f"{ctx_slug}_reals.mmap")
            preds_arr = np.memmap(preds_path, dtype=np.float32, mode="w+", shape=(ctx_count, output_dim))
            reals_arr = np.memmap(reals_path, dtype=np.float32, mode="w+", shape=(ctx_count, output_dim))

            x_hvg_arr = None
            counts_pred_arr = None
            if store_raw_expression:
                if cfg["data"]["kwargs"]["output_space"] == "gene":
                    x_hvg_shape = (ctx_count, hvg_dim)
                else:
                    x_hvg_shape = (ctx_count, gene_dim)
                x_hvg_path = os.path.join(tmp_dir.name, f"{ctx_slug}_x_hvg.mmap")
                counts_path = os.path.join(tmp_dir.name, f"{ctx_slug}_counts_pred.mmap")
                x_hvg_arr = np.memmap(x_hvg_path, dtype=np.float32, mode="w+", shape=x_hvg_shape)
                counts_pred_arr = np.memmap(counts_path, dtype=np.float32, mode="w+", shape=x_hvg_shape)

            context_store[ctx] = {
                "slug": ctx_slug,
                "offset": 0,
                "preds": preds_arr,
                "reals": reals_arr,
                "x_hvg": x_hvg_arr,
                "counts_pred": counts_pred_arr,
                "obs": {
                    "pert_name": [],
                    "celltype_name": [],
                    "batch": [],
                    "pert_cell_barcode": [],
                    "ctrl_cell_barcode": [],
                },
            }

        celltype_order: list = []
        celltype_seen = set()

        logger.info("Generating predictions on test set using split-by-context loop...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                batch_preds = model.predict_step(batch, batch_idx, padded=False)

                batch_size = batch_preds["preds"].shape[0]
                batch_labels = get_batch_labels(
                    (
                        batch.get("batch_name"),
                        batch_preds.get("batch_name"),
                        batch_preds.get("batch"),
                    ),
                    batch_size,
                )

                pert_names = ensure_list(batch_preds.get("pert_name"), batch_size)
                celltypes = ensure_list(batch_preds.get("celltype_name"), batch_size)
                if len(pert_names) != batch_size or len(celltypes) != batch_size:
                    raise ValueError("Mismatch between batch size and pert/celltype metadata lengths.")
                context_labels, detected_mode = resolve_context_labels(batch, batch_size, celltypes)
                if context_labels is None:
                    raise ValueError(
                        "Unable to resolve context labels for split-by-context. "
                        f"Expected {context_mode or 'cell_type'}."
                    )
                if detected_mode != context_mode:
                    raise ValueError(
                        f"Inconsistent context source during prediction: saw '{detected_mode}' after '{context_mode}'."
                    )

                for ct in celltypes:
                    if ct not in celltype_seen:
                        celltype_seen.add(ct)
                        celltype_order.append(ct)

                pert_barcodes = None
                ctrl_barcodes = None
                if "pert_cell_barcode" in batch_preds:
                    pert_barcodes = ensure_list(batch_preds.get("pert_cell_barcode"), batch_size)
                    ctrl_barcodes = ensure_list(batch_preds.get("ctrl_cell_barcode"), batch_size)

                batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float32)
                batch_real_np = batch_preds["pert_cell_emb"].cpu().numpy().astype(np.float32)

                batch_real_gene_np = None
                batch_gene_pred_np = None
                if store_raw_expression:
                    batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                    batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)

                label_to_indices: dict[str, list[int]] = {}
                for idx, label in enumerate(context_labels):
                    label_to_indices.setdefault(label, []).append(idx)

                for label, idxs in label_to_indices.items():
                    if label not in context_store:
                        raise ValueError(f"Encountered unseen context label '{label}' during prediction.")
                    store = context_store[label]
                    start = store["offset"]
                    end = start + len(idxs)
                    store["preds"][start:end, :] = batch_pred_np[idxs, :]
                    store["reals"][start:end, :] = batch_real_np[idxs, :]
                    if store_raw_expression:
                        store["x_hvg"][start:end, :] = batch_real_gene_np[idxs, :]
                        store["counts_pred"][start:end, :] = batch_gene_pred_np[idxs, :]
                    store["obs"]["pert_name"].extend([pert_names[i] for i in idxs])
                    store["obs"]["celltype_name"].extend([celltypes[i] for i in idxs])
                    store["obs"]["batch"].extend([batch_labels[i] for i in idxs])
                    if pert_barcodes is not None:
                        store["obs"]["pert_cell_barcode"].extend([pert_barcodes[i] for i in idxs])
                        store["obs"]["ctrl_cell_barcode"].extend([ctrl_barcodes[i] for i in idxs])
                    store["offset"] = end

        for ctx, store in context_store.items():
            if store["offset"] != context_counts[ctx]:
                logger.warning(
                    "Context '%s' count mismatch: expected %d rows, wrote %d.",
                    ctx,
                    context_counts[ctx],
                    store["offset"],
                )

        shared_perts = None
        if args.shared_only:
            try:
                shared_perts = data_module.get_shared_perturbations()
                if len(shared_perts) == 0:
                    logger.warning("No shared perturbations between train and test; skipping filtering.")
                    shared_perts = None
            except Exception as e:
                logger.warning(
                    "Failed to resolve shared perturbations (%s). Proceeding without filter.",
                    str(e),
                )

        context_adatas_pred = {}
        context_adatas_real = {}
        for ctx in context_order:
            store = context_store[ctx]
            obs_dict = {
                data_module.pert_col: store["obs"]["pert_name"],
                data_module.cell_type_key: store["obs"]["celltype_name"],
                batch_obs_key: store["obs"]["batch"],
            }
            if data_module.batch_col and data_module.batch_col != batch_obs_key:
                obs_dict[data_module.batch_col] = store["obs"]["batch"]
            if len(store["obs"]["pert_cell_barcode"]) > 0:
                obs_dict["pert_cell_barcode"] = store["obs"]["pert_cell_barcode"]
                obs_dict["ctrl_cell_barcode"] = store["obs"]["ctrl_cell_barcode"]
            obs = pd.DataFrame(obs_dict)

            if store_raw_expression:
                adata_pred = anndata.AnnData(X=store["counts_pred"], obs=obs)
                adata_real = anndata.AnnData(X=store["x_hvg"], obs=obs)
                adata_pred.obsm[data_module.embed_key] = store["preds"]
                adata_real.obsm[data_module.embed_key] = store["reals"]
            else:
                adata_pred = anndata.AnnData(X=store["preds"], obs=obs)
                adata_real = anndata.AnnData(X=store["reals"], obs=obs)

            clip_anndata_values(adata_pred, max_value=14.0)
            clip_anndata_values(adata_real, max_value=14.0)

            if shared_perts is not None and len(shared_perts) > 0:
                mask = adata_pred.obs[data_module.pert_col].isin(shared_perts)
                before_n = adata_pred.n_obs
                adata_pred = adata_pred[mask].copy()
                adata_real = adata_real[mask].copy()
                logger.info(
                    "Context '%s': filtered cells %d -> %d (shared perturbations).",
                    ctx,
                    before_n,
                    adata_pred.n_obs,
                )

            ctx_slug = store["slug"]
            adata_pred_path = os.path.join(results_dir, f"adata_pred_{ctx_slug}.h5ad")
            adata_real_path = os.path.join(results_dir, f"adata_real_{ctx_slug}.h5ad")
            adata_pred.write_h5ad(adata_pred_path)
            adata_real.write_h5ad(adata_real_path)
            logger.info("Saved adata_pred to %s", adata_pred_path)
            logger.info("Saved adata_real to %s", adata_real_path)

            context_adatas_pred[ctx] = adata_pred
            context_adatas_real[ctx] = adata_real

        if not args.predict_only:
            logger.info("Computing metrics using cell-eval...")
            control_pert = data_module.get_control_pert()

            for ct in celltype_order:
                pred_parts = []
                real_parts = []
                for ctx in context_order:
                    pred_ctx = context_adatas_pred[ctx]
                    real_ctx = context_adatas_real[ctx]
                    mask = pred_ctx.obs[data_module.cell_type_key] == ct
                    if mask.any():
                        pred_parts.append(pred_ctx[mask])
                        real_parts.append(real_ctx[mask])

                if len(pred_parts) == 0:
                    continue

                if len(pred_parts) == 1:
                    pred_ct = pred_parts[0]
                    real_ct = real_parts[0]
                else:
                    pred_ct = anndata.concat(pred_parts, merge="same")
                    real_ct = anndata.concat(real_parts, merge="same")

                pdex_kwargs = dict(exp_post_agg=True, is_log1p=True)
                evaluator = MetricsEvaluator(
                    adata_pred=pred_ct,
                    adata_real=real_ct,
                    control_pert=control_pert,
                    pert_col=data_module.pert_col,
                    outdir=results_dir,
                    prefix=ct,
                    pdex_kwargs=pdex_kwargs,
                    batch_size=2048,
                )

                evaluator.compute(
                    profile=args.profile,
                    metric_configs={
                        "discrimination_score": {
                            "embed_key": data_module.embed_key,
                        }
                        if data_module.embed_key and data_module.embed_key != "X_hvg"
                        else {},
                        "pearson_edistance": {
                            "embed_key": data_module.embed_key,
                            "n_jobs": -1,  # set to all available cores
                        }
                        if data_module.embed_key and data_module.embed_key != "X_hvg"
                        else {
                            "n_jobs": -1,
                        },
                    }
                    if data_module.embed_key and data_module.embed_key != "X_hvg"
                    else {},
                    skip_metrics=["pearson_edistance", "clustering_agreement"],
                )

        tmp_dir.cleanup()
        return

    final_preds = np.empty((num_cells, output_dim), dtype=np.float32)
    final_reals = np.empty((num_cells, output_dim), dtype=np.float32)

    final_X_hvg = None
    final_pert_cell_counts_preds = None
    if store_raw_expression:
        # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
        if cfg["data"]["kwargs"]["output_space"] == "gene":
            final_X_hvg = np.empty((num_cells, hvg_dim), dtype=np.float32)
            final_pert_cell_counts_preds = np.empty((num_cells, hvg_dim), dtype=np.float32)
        if cfg["data"]["kwargs"]["output_space"] == "all":
            final_X_hvg = np.empty((num_cells, gene_dim), dtype=np.float32)
            final_pert_cell_counts_preds = np.empty((num_cells, gene_dim), dtype=np.float32)

    current_idx = 0

    # Initialize aggregation variables directly
    all_pert_names = []
    all_celltypes = []
    all_gem_groups = []
    all_pert_barcodes = []
    all_ctrl_barcodes = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
            # Move each tensor in the batch to the model's device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Get predictions
            batch_preds = model.predict_step(batch, batch_idx, padded=False)

            # Extract metadata and data directly from batch_preds
            # Handle pert_name
            if isinstance(batch_preds["pert_name"], list):
                all_pert_names.extend(batch_preds["pert_name"])
            else:
                all_pert_names.append(batch_preds["pert_name"])

            if "pert_cell_barcode" in batch_preds:
                if isinstance(batch_preds["pert_cell_barcode"], list):
                    all_pert_barcodes.extend(batch_preds["pert_cell_barcode"])
                    all_ctrl_barcodes.extend(batch_preds["ctrl_cell_barcode"])
                else:
                    all_pert_barcodes.append(batch_preds["pert_cell_barcode"])
                    all_ctrl_barcodes.append(batch_preds["ctrl_cell_barcode"])

            # Handle celltype_name
            if isinstance(batch_preds["celltype_name"], list):
                all_celltypes.extend(batch_preds["celltype_name"])
            else:
                all_celltypes.append(batch_preds["celltype_name"])

            batch_size = batch_preds["preds"].shape[0]

            # Handle gem_group - prefer human-readable batch names when available
            batch_labels = get_batch_labels(
                (
                    batch.get("batch_name"),
                    batch_preds.get("batch_name"),
                    batch_preds.get("batch"),
                ),
                batch_size,
            )
            all_gem_groups.extend(batch_labels)

            batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float32)
            batch_real_np = batch_preds["pert_cell_emb"].cpu().numpy().astype(np.float32)
            final_preds[current_idx : current_idx + batch_size, :] = batch_pred_np
            final_reals[current_idx : current_idx + batch_size, :] = batch_real_np
            current_idx += batch_size

            # Handle X_hvg for HVG space ground truth
            if final_X_hvg is not None:
                batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                final_X_hvg[current_idx - batch_size : current_idx, :] = batch_real_gene_np

            # Handle decoded gene predictions if available
            if final_pert_cell_counts_preds is not None:
                batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)
                final_pert_cell_counts_preds[current_idx - batch_size : current_idx, :] = batch_gene_pred_np

    logger.info("Creating anndatas from predictions from manual loop...")

    # Build pandas DataFrame for obs and var
    print(batch_obs_key)
    df_dict = {
        data_module.pert_col: all_pert_names,
        data_module.cell_type_key: all_celltypes,
        batch_obs_key: all_gem_groups,
    }
    if data_module.batch_col and data_module.batch_col != batch_obs_key:
        print("\t\t STORING BATCH")
        df_dict[data_module.batch_col] = all_gem_groups

    if len(all_pert_barcodes) > 0:
        df_dict["pert_cell_barcode"] = all_pert_barcodes
        df_dict["ctrl_cell_barcode"] = all_ctrl_barcodes

    obs = pd.DataFrame(df_dict)

    gene_names = var_dims["gene_names"]
    var = pd.DataFrame({"gene_names": gene_names})

    if final_X_hvg is not None:
        # if len(gene_names) != final_pert_cell_counts_preds.shape[1]:
        #     gene_names = np.load(
        #         "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
        #     )
        #     var = pd.DataFrame({"gene_names": gene_names})

        # Create adata for predictions - using the decoded gene expression values
        adata_pred = anndata.AnnData(X=final_pert_cell_counts_preds, obs=obs)
        # Create adata for real - using the true gene expression values
        adata_real = anndata.AnnData(X=final_X_hvg, obs=obs)

        # add the embedding predictions
        adata_pred.obsm[data_module.embed_key] = final_preds
        adata_real.obsm[data_module.embed_key] = final_reals
        logger.info(f"Added predicted embeddings to adata.obsm['{data_module.embed_key}']")
    else:
        # if len(gene_names) != final_preds.shape[1]:
        #     gene_names = np.load(
        #         "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
        #     )
        #     var = pd.DataFrame({"gene_names": gene_names})

        # Create adata for predictions - model was trained on gene expression space already
        # adata_pred = anndata.AnnData(X=final_preds, obs=obs, var=var)
        adata_pred = anndata.AnnData(X=final_preds, obs=obs)
        # Create adata for real - using the true gene expression values
        # adata_real = anndata.AnnData(X=final_reals, obs=obs, var=var)
        adata_real = anndata.AnnData(X=final_reals, obs=obs)

    # Clip extreme values to keep cell-eval log1p checks happy.
    clip_anndata_values(adata_pred, max_value=14.0)
    clip_anndata_values(adata_real, max_value=14.0)
    logger.info("Clipped adata_pred and adata_real X values to [0.0, 14.0] before evaluation.")

    # Optionally filter to perturbations seen in at least one training context
    if args.shared_only:
        try:
            shared_perts = data_module.get_shared_perturbations()
            if len(shared_perts) == 0:
                logger.warning("No shared perturbations between train and test; skipping filtering.")
            else:
                logger.info(
                    "Filtering to %d shared perturbations present in train ∩ test.",
                    len(shared_perts),
                )
                mask = adata_pred.obs[data_module.pert_col].isin(shared_perts)
                before_n = adata_pred.n_obs
                adata_pred = adata_pred[mask].copy()
                adata_real = adata_real[mask].copy()
                logger.info(
                    "Filtered cells: %d -> %d (kept only seen perturbations)",
                    before_n,
                    adata_pred.n_obs,
                )
        except Exception as e:
            logger.warning(
                "Failed to filter by shared perturbations (%s). Proceeding without filter.",
                str(e),
            )

    # Save the AnnData objects
    if args.eval_train_data:
        results_dir = os.path.join(args.output_dir, "eval_train_" + os.path.basename(args.checkpoint))
    else:
        results_dir = os.path.join(args.output_dir, "eval_" + os.path.basename(args.checkpoint))
    os.makedirs(results_dir, exist_ok=True)
    adata_pred_path = os.path.join(results_dir, "adata_pred.h5ad")
    adata_real_path = os.path.join(results_dir, "adata_real.h5ad")

    adata_pred.write_h5ad(adata_pred_path)
    adata_real.write_h5ad(adata_real_path)

    logger.info(f"Saved adata_pred to {adata_pred_path}")
    logger.info(f"Saved adata_real to {adata_real_path}")

    if not args.predict_only:
        # 6. Compute metrics using cell-eval
        logger.info("Computing metrics using cell-eval...")

        control_pert = data_module.get_control_pert()

        ct_split_real = split_anndata_on_celltype(adata=adata_real, celltype_col=data_module.cell_type_key)
        ct_split_pred = split_anndata_on_celltype(adata=adata_pred, celltype_col=data_module.cell_type_key)

        assert len(ct_split_real) == len(ct_split_pred), (
            f"Number of celltypes in real and pred anndata must match: {len(ct_split_real)} != {len(ct_split_pred)}"
        )

        pdex_kwargs = dict(exp_post_agg=True, is_log1p=True)

        for ct in ct_split_real.keys():
            real_ct = ct_split_real[ct]
            pred_ct = ct_split_pred[ct]

            evaluator = MetricsEvaluator(
                adata_pred=pred_ct,
                adata_real=real_ct,
                control_pert=control_pert,
                pert_col=data_module.pert_col,
                outdir=results_dir,
                prefix=ct,
                pdex_kwargs=pdex_kwargs,
                batch_size=2048,
            )

            evaluator.compute(
                profile=args.profile,
                metric_configs={
                    "discrimination_score": {
                        "embed_key": data_module.embed_key,
                    }
                    if data_module.embed_key and data_module.embed_key != "X_hvg"
                    else {},
                    "pearson_edistance": {
                        "embed_key": data_module.embed_key,
                        "n_jobs": -1,  # set to all available cores
                    }
                    if data_module.embed_key and data_module.embed_key != "X_hvg"
                    else {
                        "n_jobs": -1,
                    },
                }
                if data_module.embed_key and data_module.embed_key != "X_hvg"
                else {},
                skip_metrics=["pearson_edistance", "clustering_agreement"],
            )
