# Hyperparameters, System Configurations & File Dependencies (`v41`)

## 1. Global System Parameters

| Parameter | Symbol / Variable | Value | Description |
| :--- | :--- | :--- | :--- |
| **Total Evaluation Batch** | $N_{\text{eval}}$ | `35,665` | Evaluation image count ($20,097$ test + $15,568$ unseen). |
| **Total Candidate Classes** | $N_{\text{cls}}$ | `17,393` | Full competition label space (`data/dl/all_classes.pkl`). |
| **Seen Training Classes** | $S$ | `5,795` | Class count in training labels (`data/dl/label_train.json`). |
| **Unseen Candidate Classes** | $U$ | `11,598` | Novel species classes ($N_{\text{cls}} - S$). |
| **Gating Split Fraction** | $f_{\text{seen}}$ | `0.72` | Fraction of eval images routed to seen classifier head ($25,679$ images). |
| **Gating Text Margin Weight** | $w_{\text{margin}}$ | `2.0` | Weight of `text_margin` relative to `img_seenmax` in score $G(x)$. |
| **Taxonomy Vector Lambda** | $\lambda$ | `4.0` | Weight of `TseenTax` in seen prototype matching. |
| **Photo-Bank Maxpool TOPM** | `TOPM` | `4` | Top-$k$ photos averaged per unseen candidate species in maxpool leg. |
| **Photo-Bank Image Weight** | $w_{\text{img}}$ | `4.0` / `3.0` | Weight of `ctftshift` photo-bank maxpool leg in unseen route score. |
| **BioCLIP-2 Proto Weight** | $B2\_W$ | `2.5` | Weight of `bioclip-2` (ViT-L/14) prototype leg in unseen route score. |
| **TaxaBind Unseen Weight** | $w_{\text{tb}}$ | `1.0` | Weight of TaxaBind text logit in unseen route ensemble. |
| **Dual-Softmax Temp (Col)** | $\tau_{\text{col}}$ | `0.05` | Sample-wise normalization temperature in `dbnorm`. |
| **Dual-Softmax Temp (Row)** | $\tau_{\text{row}}$ | `0.50` | Class-wise normalization temperature in `dbnorm`. |
| **Sinkhorn Temperature** | $\tau_{\text{sink}}$ | `2.0` | Softmax temperature for Sinkhorn optimal transport. |
| **Sinkhorn Iterations** | $N_{\text{iter}}$ | `50` | Scaling iteration count in Sinkhorn algorithm. |
| **Strict Non-Trans. Threshold** | $\text{THR}$ | `-2.0423` | Frozen absolute scalar gating threshold (`v31`). |

---

## 2. Model Member Weights & Configurations

| Member Key | Underlying Model Architecture | Training Condition / Checkpoint | Member Weight | Taxonomy Context |
| :--- | :--- | :--- | :--- | :--- |
| `ctftshift` | `imageomics/bioclip-2.5-vith14` | Shift-aug LoRA (`scale=(0.35,1.0), ratio=(0.5,2.0)`) | `1.0` | Enabled |
| `ftshift` | `imageomics/bioclip-2.5-vith14` | Full-FT 224px shift retrain | `2.5` | Enabled |
| `fullft336shift` | `imageomics/bioclip-2.5-vith14` | Full-FT 336px shift retrain | `2.5` | Enabled |
| `L` | `imageomics/bioclip-2` (ViT-L/14) | Frozen baseline | `0.0` | Text leg only ($0.5 \cdot \text{name}$) |
| `fullft336_v2` | `imageomics/bioclip-2.5-vith14` | 336px full-FT v2 | `0.0` | Text leg only ($0.75 \cdot \text{taxon}$) |
| `taxabind` | `MVRL/taxabind-vit-b-16` | Frozen general biodiversity dual encoder | `1.0` | Text leg only ($1.0 \cdot \text{taxctx}$) |
| `bioclip2` | `imageomics/bioclip-2` (ViT-L/14) | Frozen baseline | `2.5` | Unseen image proto leg ($B2\_W=2.5$) |

---

## 3. Training & LoRA Hyperparameters (`src/contrastive_ft.py`)

| Setting | Hyperparameter Flag | Value | Description |
| :--- | :--- | :--- | :--- |
| **Base Model** | `--model` | `hf-hub:imageomics/bioclip-2.5-vith14` | BioCLIP 2.5 ViT-H/14 foundation backbone. |
| **LoRA Rank** | `--rank` | `16` | LoRA bottleneck rank $r$. |
| **LoRA Alpha** | `--alpha` | `32` | LoRA scaling parameter $\alpha$ ($\frac{\alpha}{r} = 2.0$). |
| **LoRA Dropout** | `--dropout` | `0.05` | Dropout probability applied to LoRA input. |
| **Top Blocks Adapted**| `--top_k` | `12` | Top $12$ residual blocks adapted out of $32$. |
| **Batch Size** | `--bs` | `128` | Training mini-batch size. |
| **Learning Rate** | `--lr` | `5e-4` | Learning rate for LoRA $A$ and $B$ parameters. |
| **Weight Decay** | `--weight_decay` | `0.0` | L2 weight regularization penalty. |
| **Max Training Steps**| `--max_steps` | `800` | Total optimization steps. |
| **InfoNCE Temperature**| `--temp` | `30.0` | Temperature scale for contrastive loss. |
| **Shift Aug Scale** | `scale` | `(0.35, 1.0)` | RandomResizedCrop scale range. |
| **Shift Aug Ratio** | `ratio` | `(0.5, 2.0)` | RandomResizedCrop aspect ratio range. |

---

## 4. Primary Input File Dependencies (`v41`)

| Relative File Path | Role & Content |
| :--- | :--- |
| `data/dl/all_classes.pkl` | Full competition species list ($17,393$ entries). |
| `data/dl/label_train.json` | Mapping of training image filenames to seen species labels ($99,979$ training images). |
| `outputs/text_emb_h_taxon.pt` | BioCLIP-H text embeddings for scientific taxon names. |
| `outputs/text_emb_h_promptens.pt` | BioCLIP-H text embeddings for taxonomy context prompts (`emb_taxctx`). |
| `outputs/text_emb_taxabind_taxctx.pt` | TaxaBind text embeddings for taxonomy context prompts. |
| `outputs/inat_photo_bank_ctftshift.pt` | Multi-photo feature bank extracted from iNaturalist S3 photos using `ctftshift` for maxpooling (`TOPM=4`). |
| `outputs/inat_protos_bioclip2.pt` | Visual prototype feature vectors extracted using frozen `BioCLIP-2` (ViT-L/14). |
| `outputs/emb_test_ctftshift.pt` | Feature embeddings for test set images extracted with `ctftshift`. |
| `outputs/emb_unseen_ctftshift.pt` | Feature embeddings for unseen set images extracted with `ctftshift`. |
| `outputs/emb_test_taxabind.pt` | Feature embeddings for test set images extracted with `taxabind`. |
| `outputs/emb_unseen_taxabind.pt` | Feature embeddings for unseen set images extracted with `taxabind`. |
| `outputs/emb_test_bioclip2.pt` | Feature embeddings for test set images extracted with `bioclip2` (ViT-L/14). |
| `outputs/emb_unseen_bioclip2.pt` | Feature embeddings for unseen set images extracted with `bioclip2` (ViT-L/14). |

---

## 5. Execution Environment & Dependencies

- **Python Version**: `3.10+`
- **PyTorch Version**: `2.1+` with CUDA 12.1 support
- **Key Python Packages**: `open_clip_torch`, `torchvision`, `pillow`, `scipy`, `numpy`
- **Environment Activation**: `conda activate onet`
