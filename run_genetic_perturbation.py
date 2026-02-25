"""
Genetic perturbation prediction using STATE (ST-SE-Replogle).

Two-step pipeline:
  1. Embed cells using SE-600M
  2. Predict perturbed transcriptome using ST-SE-Replogle

Prerequisites:
  - Install STATE: `uv tool install arc-state` or `uv pip install arc-state`
  - Download models (see DOWNLOAD COMMANDS below)

DOWNLOAD COMMANDS (run once):
  ```
  # Install git-lfs first if needed
  git lfs install

  # 1) SE-600M (cell embedding model, ~11.5 GB checkpoint)
  git clone https://huggingface.co/arcinstitute/SE-600M ./models/SE-600M

  # 2) ST-SE-Replogle (perturbation prediction model)
  git clone https://huggingface.co/arcinstitute/ST-SE-Replogle ./models/ST-SE-Replogle
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


def step2_predict_perturbation(
    embedded_h5ad: str,
    st_model_dir: str,
    target_gene: str,
    output_h5ad: str,
    embed_key: str = "X_state",
    control_pert: str = "non-targeting",
    pert_col: str = "target_gene",
):
    """Predict perturbed transcriptome using ST-SE-Replogle."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    print("=" * 60)
    print(f"Step 2: Predicting perturbation for gene '{target_gene}'")
    print("=" * 60)

    # Load model artifacts
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

    # Load checkpoint
    ckpt_dir = os.path.join(st_model_dir, "checkpoints")
    ckpt_candidates = ["final.ckpt", "last.ckpt", "best.ckpt"]
    checkpoint = None
    if os.path.isdir(ckpt_dir):
        for name in ckpt_candidates:
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
    print(f"Using ST checkpoint: {checkpoint}")

    model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint)
    model.eval()
    device = next(model.parameters()).device
    cell_set_len = getattr(model, "cell_sentence_len", 256)
    uses_batch_encoder = getattr(model, "batch_encoder", None) is not None

    # Check if target gene exists in the perturbation map
    if target_gene not in pert_names_in_map:
        print(f"\nWARNING: '{target_gene}' not found in perturbation map!")
        print(f"Available perturbations ({len(pert_names_in_map)} total):")
        matches = [p for p in pert_names_in_map if target_gene.upper() in p.upper()]
        if matches:
            print(f"  Partial matches: {matches[:20]}")
        else:
            print(f"  First 30: {pert_names_in_map[:30]}")
        print("Using control fallback vector (results may be meaningless).")

    # Load embedded adata
    adata = sc.read_h5ad(embedded_h5ad)
    print(f"Loaded {adata.n_obs} cells from {embedded_h5ad}")

    # Prepare: all cells are treated as controls, we simulate the target perturbation
    if embed_key in adata.obsm:
        X_in = np.asarray(adata.obsm[embed_key])
    else:
        raise KeyError(f"Embedding key '{embed_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}")

    # Get perturbation one-hot vector
    pert_vec = pert_onehot_map.get(target_gene, None)
    if pert_vec is None:
        pert_vec = torch.zeros(pert_dim, dtype=torch.float32)
        if pert_dim and pert_dim > 0:
            pert_vec[0] = 1.0

    # Run inference
    n_cells = X_in.shape[0]
    sim_out = np.zeros_like(X_in, dtype=np.float32)

    print(f"Running forward pass on {n_cells} cells (set length: {cell_set_len})...")

    with torch.no_grad():
        start = 0
        while start < n_cells:
            end = min(start + cell_set_len, n_cells)
            win_size = end - start

            ctrl_basal = X_in[start:end]
            X_batch = torch.tensor(ctrl_basal, dtype=torch.float32, device=device)
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
            sim_out[start:end] = preds

            start = end

    # Save results
    adata_out = adata.copy()
    adata_out.obsm[f"{embed_key}_perturbed"] = sim_out
    if pert_col not in adata_out.obs.columns:
        adata_out.obs[pert_col] = target_gene
    adata_out.write_h5ad(output_h5ad)

    print(f"\nResults saved to {output_h5ad}")
    print(f"  Original embeddings:  .obsm['{embed_key}']")
    print(f"  Perturbed embeddings: .obsm['{embed_key}_perturbed']")
    print(f"  Delta (perturbation effect) can be computed as:")
    print(f"    delta = adata.obsm['{embed_key}_perturbed'] - adata.obsm['{embed_key}']")

    return output_h5ad


def main():
    parser = argparse.ArgumentParser(
        description="Predict gene knockdown effects using STATE (SE-600M + ST-SE-Replogle)"
    )
    parser.add_argument("--input", required=True, help="Input h5ad file (raw or normalized)")
    parser.add_argument("--gene", required=True, help="Target gene to knock down (e.g., TP53, KRAS)")
    parser.add_argument(
        "--cell-type",
        default="k562",
        choices=["k562", "rpe1", "jurkat", "hepg2"],
        help="Cell type model to use (default: k562)",
    )
    parser.add_argument(
        "--task",
        default="fewshot",
        choices=["fewshot", "zeroshot"],
        help="Model variant (default: fewshot)",
    )
    parser.add_argument("--se-model-dir", default="./models/SE-600M", help="Path to SE-600M model directory")
    parser.add_argument(
        "--st-model-dir",
        default="./models/ST-SE-Replogle",
        help="Path to ST-SE-Replogle model directory",
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

    # Resolve ST model directory to the specific cell type run
    st_run_dir = os.path.join(args.st_model_dir, args.task, args.cell_type)
    if not os.path.isdir(st_run_dir):
        print(f"ERROR: ST model directory not found: {st_run_dir}")
        print(f"Available runs in {args.st_model_dir}:")
        for task_dir in ["fewshot", "zeroshot"]:
            td = os.path.join(args.st_model_dir, task_dir)
            if os.path.isdir(td):
                print(f"  {task_dir}/: {os.listdir(td)}")
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

    # Step 2: Predict perturbation
    step2_predict_perturbation(
        embedded_h5ad=embedded_path,
        st_model_dir=st_run_dir,
        target_gene=args.gene,
        output_h5ad=output_path,
        embed_key=args.embed_key,
    )

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nTo analyze results in Python:")
    print(f"  import scanpy as sc")
    print(f"  adata = sc.read_h5ad('{output_path}')")
    print(f"  original = adata.obsm['{args.embed_key}']")
    print(f"  perturbed = adata.obsm['{args.embed_key}_perturbed']")
    print(f"  delta = perturbed - original  # perturbation effect in embedding space")


if __name__ == "__main__":
    main()
