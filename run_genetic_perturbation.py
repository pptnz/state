"""
Genetic perturbation prediction using STATE (SE-600M + ST-SE-Replogle-full).

Two-step pipeline:
  1. Embed cells using SE-600M (raw expression → 2058-dim cell embedding)
  2. Predict perturbed embeddings + ALL-gene count-level expression using
     ST-SE-Replogle-full (output_space=all, gene_decoder outputs ~36k genes)

Prerequisites:
  - Install STATE: `uv tool install arc-state` or `uv pip install arc-state`
  - Download models (see DOWNLOAD COMMANDS below)

DOWNLOAD COMMANDS (run once):
    ```
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="arcinstitute/SE-600M",
        local_dir="./models/SE-600M",
        local_dir_use_symlinks=False
    )

    snapshot_download(
        repo_id="arcinstitute/st-se-replogle-full",
        local_dir="./models/ST-SE-Replogle-full",
        local_dir_use_symlinks=False
    )
    ```

Usage:
  python run_genetic_perturbation.py \
    --input /path/to/your_cells.h5ad \
    --gene TP53 \
    --cell-type k562
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch


def step1_embed_cells(input_h5ad: str, se_model_dir: str, output_h5ad: str, embed_key: str = "X_state"):
    """Embed cells using SE-600M to produce cell embeddings."""
    from state.emb.inference import Inference

    print("=" * 60)
    print("Step 1: Embedding cells with SE-600M")
    print("=" * 60)

    checkpoint = os.path.join(se_model_dir, "se600m_epoch16.ckpt")
    if not os.path.exists(checkpoint):
        ckpts = list(Path(se_model_dir).glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files found in {se_model_dir}")
        checkpoint = str(ckpts[-1])
        print(f"Using checkpoint: {checkpoint}")

    pe_path = os.path.join(se_model_dir, "protein_embeddings.pt")
    protein_embeds = None
    if os.path.exists(pe_path):
        protein_embeds = torch.load(pe_path, weights_only=False, map_location="cpu")

    inferer = Inference(protein_embeds=protein_embeds)
    inferer.load_model(checkpoint)

    inferer.encode_adata(
        input_adata_path=input_h5ad,
        output_adata_path=output_h5ad,
        emb_key=embed_key,
    )
    print(f"Embeddings saved to {output_h5ad} under .obsm['{embed_key}']")
    return output_h5ad


def _find_st_checkpoint(st_model_dir: str) -> str:
    """Locate a checkpoint file inside *st_model_dir*."""
    ckpt_dir = os.path.join(st_model_dir, "checkpoints")
    candidates = ["final.ckpt", "last.ckpt", "best.ckpt"]
    checkpoint = None
    if os.path.isdir(ckpt_dir):
        for name in candidates:
            p = os.path.join(ckpt_dir, name)
            if os.path.exists(p):
                checkpoint = p
                break
        if checkpoint is None:
            ckpts = list(Path(ckpt_dir).glob("*.ckpt"))
            if ckpts:
                checkpoint = str(ckpts[-1])
    if checkpoint is None:
        for name in ["eval_best.ckpt", "eval_last.ckpt"]:
            p = os.path.join(st_model_dir, name)
            if os.path.exists(p):
                checkpoint = p
                break
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {st_model_dir}")
    return checkpoint


def step2_predict_perturbation(
    embedded_h5ad: str,
    st_model_dir: str,
    target_gene: str,
    output_h5ad: str,
    embed_key: str = "X_state",
    control_pert: str = "non-targeting",
    pert_col: str = "target_gene",
):
    """Predict perturbed transcriptome and decode to gene counts when possible."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    print("=" * 60)
    print(f"Step 2: Predicting perturbation for gene '{target_gene}'")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load model artifacts
    # ------------------------------------------------------------------
    import yaml

    config_path = os.path.join(st_model_dir, "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    var_dims_path = os.path.join(st_model_dir, "var_dims.pkl")
    with open(var_dims_path, "rb") as f:
        var_dims = pickle.load(f)

    pert_onehot_map = torch.load(
        os.path.join(st_model_dir, "pert_onehot_map.pt"), weights_only=False
    )
    pert_names_in_map = [str(k) for k in pert_onehot_map.keys()]
    pert_dim = var_dims.get("pert_dim")

    checkpoint = _find_st_checkpoint(st_model_dir)
    print(f"Using ST checkpoint: {checkpoint}")

    model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint)
    model.eval()
    device = next(model.parameters()).device
    cell_set_len = getattr(model, "cell_sentence_len", 256)
    uses_batch_encoder = getattr(model, "batch_encoder", None) is not None

    has_gene_decoder = getattr(model, "gene_decoder", None) is not None
    gene_names = var_dims.get("gene_names", None)
    if has_gene_decoder:
        n_decoder_genes = model.gene_decoder.gene_dim()
        print(f"Gene decoder found — will decode to {n_decoder_genes} genes (count-level)")
    else:
        print("WARNING: No gene_decoder in this checkpoint (output_space='embedding').")
        print("  Count-level prediction unavailable; only embedding-space results will be saved.")

    # ------------------------------------------------------------------
    # Validate target gene
    # ------------------------------------------------------------------
    if target_gene not in pert_names_in_map:
        print(f"\nWARNING: '{target_gene}' not found in perturbation map!")
        print(f"Available perturbations ({len(pert_names_in_map)} total):")
        matches = [p for p in pert_names_in_map if target_gene.upper() in p.upper()]
        if matches:
            print(f"  Partial matches: {matches[:20]}")
        else:
            print(f"  First 30: {pert_names_in_map[:30]}")
        print("Using control fallback vector (results may be meaningless).")

    # ------------------------------------------------------------------
    # Load embedded adata
    # ------------------------------------------------------------------
    adata = sc.read_h5ad(embedded_h5ad)
    print(f"Loaded {adata.n_obs} cells from {embedded_h5ad}")

    if embed_key in adata.obsm:
        X_in = np.asarray(adata.obsm[embed_key])
    else:
        raise KeyError(f"Embedding key '{embed_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}")

    pert_vec = pert_onehot_map.get(target_gene, None)
    if pert_vec is None:
        pert_vec = torch.zeros(pert_dim, dtype=torch.float32)
        if pert_dim and pert_dim > 0:
            pert_vec[0] = 1.0

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    n_cells = X_in.shape[0]
    emb_out = np.zeros_like(X_in, dtype=np.float32)

    if has_gene_decoder:
        n_decoder_genes = model.gene_decoder.gene_dim()
        pert_counts = np.zeros((n_cells, n_decoder_genes), dtype=np.float32)
        ctrl_counts = np.zeros((n_cells, n_decoder_genes), dtype=np.float32)

    print(f"Running forward pass on {n_cells} cells (set length: {cell_set_len})...")

    with torch.no_grad():
        start = 0
        while start < n_cells:
            end = min(start + cell_set_len, n_cells)
            win_size = end - start

            X_batch = torch.tensor(X_in[start:end], dtype=torch.float32, device=device)
            pert_oh = pert_vec.float().unsqueeze(0).repeat(win_size, 1).to(device)

            batch = {
                "ctrl_cell_emb": X_batch,
                "pert_emb": pert_oh,
                "pert_name": [target_gene] * win_size,
            }
            if uses_batch_encoder:
                batch["batch"] = torch.zeros(win_size, dtype=torch.long, device=device)

            batch_out = model.predict_step(batch, batch_idx=0, padded=False)

            preds = batch_out["preds"].detach().cpu().numpy().astype(np.float32)
            emb_out[start:end] = preds

            if has_gene_decoder:
                # Perturbed gene counts from predict_step
                pert_pred = batch_out["pert_cell_counts_preds"]
                if pert_pred.dim() == 3:
                    pert_pred = pert_pred.reshape(-1, n_decoder_genes)
                pert_counts[start:end] = pert_pred.detach().cpu().numpy().astype(np.float32)

                # Control gene counts by decoding ctrl embeddings
                ctrl_pred = model.gene_decoder(X_batch)
                if ctrl_pred.dim() == 3:
                    ctrl_pred = ctrl_pred.reshape(-1, n_decoder_genes)
                ctrl_counts[start:end] = ctrl_pred.detach().cpu().numpy().astype(np.float32)

            start = end

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    adata_out = adata.copy()
    adata_out.obsm[f"{embed_key}_perturbed"] = emb_out
    if pert_col not in adata_out.obs.columns:
        adata_out.obs[pert_col] = target_gene

    if has_gene_decoder:
        adata_out.obsm["gene_counts_ctrl"] = ctrl_counts
        adata_out.obsm["gene_counts_pert"] = pert_counts
        adata_out.obsm["gene_counts_delta"] = pert_counts - ctrl_counts
        if gene_names is not None:
            adata_out.uns["gene_decoder_names"] = list(gene_names)

    adata_out.write_h5ad(output_h5ad)

    print(f"\nResults saved to {output_h5ad}")
    print(f"  Embeddings:  .obsm['{embed_key}'], .obsm['{embed_key}_perturbed']")
    if has_gene_decoder:
        print(f"  Gene counts: .obsm['gene_counts_ctrl']   ({ctrl_counts.shape})")
        print(f"               .obsm['gene_counts_pert']   ({pert_counts.shape})")
        print(f"               .obsm['gene_counts_delta']  (pert - ctrl)")
        print(f"  Gene names:  .uns['gene_decoder_names']  ({len(gene_names)} genes)")

    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return output_h5ad, has_gene_decoder


def main():
    parser = argparse.ArgumentParser(
        description="Predict gene knockdown effects using STATE (SE-600M + ST-SE-Replogle-full)"
    )
    parser.add_argument("--input", required=True, help="Input h5ad file (raw or normalized)")
    parser.add_argument("--gene", required=True, help="Target gene to knock down (e.g., TP53, KRAS)")
    parser.add_argument(
        "--cell-type",
        default="k562",
        choices=["k562", "rpe1", "jurkat", "hepg2"],
        help="Cell type model to use (default: k562)",
    )
    parser.add_argument("--se-model-dir", default="./models/SE-600M", help="Path to SE-600M model directory")
    parser.add_argument(
        "--st-model-dir",
        default="./models/ST-SE-Replogle-full",
        help="Path to ST-SE-Replogle-full model directory",
    )
    parser.add_argument("--output", default=None, help="Output h5ad path (default: <input>_perturbed_<gene>.h5ad)")
    parser.add_argument(
        "--embed-key",
        default="X_state",
        help="Embedding key in obsm (default: X_state)",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip SE embedding step (if input already has embeddings in --embed-key)",
    )

    args = parser.parse_args()

    # ST-SE-Replogle-full uses {cell_type}_0.99 directory layout
    st_run_dir = os.path.join(args.st_model_dir, f"{args.cell_type}_0.99")
    if not os.path.isdir(st_run_dir):
        # Fallback: try without _0.99 suffix
        st_run_dir_alt = os.path.join(args.st_model_dir, args.cell_type)
        if os.path.isdir(st_run_dir_alt):
            st_run_dir = st_run_dir_alt
        else:
            print(f"ERROR: ST model directory not found: {st_run_dir}")
            print(f"Available directories in {args.st_model_dir}:")
            if os.path.isdir(args.st_model_dir):
                print(f"  {os.listdir(args.st_model_dir)}")
            sys.exit(1)

    output_path = args.output or args.input.replace(".h5ad", f"_perturbed_{args.gene}.h5ad")
    embedded_path = args.input.replace(".h5ad", "_embedded.h5ad")

    # Step 1: Embed cells
    if args.skip_embedding:
        print(f"Skipping embedding step, using existing embeddings from {args.input}")
        embedded_path = args.input
    elif not os.path.exists(embedded_path):
        step1_embed_cells(
            input_h5ad=args.input,
            se_model_dir=args.se_model_dir,
            output_h5ad=embedded_path,
            embed_key=args.embed_key,
        )

    # Step 2: Predict perturbation + all-gene counts via gene_decoder
    _, has_gene_decoder = step2_predict_perturbation(
        embedded_h5ad=embedded_path,
        st_model_dir=st_run_dir,
        target_gene=args.gene,
        output_h5ad=output_path,
        embed_key=args.embed_key,
    )

    # ------------------------------------------------------------------
    # Export results as TSV
    # ------------------------------------------------------------------
    adata = sc.read_h5ad(output_path)

    if has_gene_decoder:
        gene_names = adata.uns["gene_decoder_names"]
        cell_ids = adata.obs.index

        gene_counts_ctrl = pd.DataFrame(
            adata.obsm["gene_counts_ctrl"].T, index=gene_names, columns=cell_ids,
        )
        gene_counts_ctrl.to_csv(args.input.replace(".h5ad", "_ctrl_" + args.gene + ".txt"), sep="\t")

        gene_counts_pert = pd.DataFrame(
            adata.obsm["gene_counts_pert"].T, index=gene_names, columns=cell_ids,
        )
        gene_counts_pert.to_csv(args.input.replace(".h5ad", "_pert_" + args.gene + ".txt"), sep="\t")

        # save adata.obs
        adata.obs.to_csv(args.input.replace(".h5ad", "_obs.csv"))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"\nTo analyze results in Python:")
    print(f"  import scanpy as sc")
    print(f"  adata = sc.read_h5ad('{output_path}')")
    if has_gene_decoder:
        n_genes = len(adata.uns["gene_decoder_names"])
        print(f"\n  # All-gene count predictions (n_cells x {n_genes} genes):")
        print(f"  ctrl  = adata.obsm['gene_counts_ctrl']")
        print(f"  pert  = adata.obsm['gene_counts_pert']")
        print(f"  delta = adata.obsm['gene_counts_delta']")
        print(f"  genes = adata.uns['gene_decoder_names']")
    print(f"\n  # Cell embeddings (n_cells x 2058):")
    print(f"  ctrl_emb = adata.obsm['{args.embed_key}']")
    print(f"  pert_emb = adata.obsm['{args.embed_key}_perturbed']")

    if has_gene_decoder:
        print(f"\nExported TSV files:")
        print(f"  {args.input.replace('.h5ad', '_ctrl_' + args.gene + '.txt')}")
        print(f"  {args.input.replace('.h5ad', '_pert_' + args.gene + '.txt')}")


if __name__ == "__main__":
    main()
