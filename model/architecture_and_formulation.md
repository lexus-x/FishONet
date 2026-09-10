# Technical Architecture & Mathematical Formulation (`v41`)

This document provides the complete, self-contained mathematical formulation, multi-modal encoder specifications, and algorithmic dataflow for the **Open-Set Species Recognition System (`v41`)**.

---

## 1. Multi-Modal Vision & Text Backbones

The `v41` system incorporates three foundation model backbones:

### 1.1 `imageomics/bioclip-2.5-vith14` (Primary Vision & Text Tower)
- **Architecture**: ViT-H/14 vision transformer ($14 \times 14$ patch size, embedding dimension $D = 1024$).
- **Pretraining**: TreeOfLife-10M dataset covering $10$ million biological images across diverse taxa.
- **Domain Adaptation**: Fine-tuned on competition training images using **Aspect-Shift Matched LoRA Adaptation**.

### 1.2 `MVRL/taxabind-vit-b-16` (Secondary Biodiversity Encoder)
- **Architecture**: ViT-B/16 dual encoder ($D_{\text{tb}} = 512$).
- **Role**: Provides an orthogonal feature representation for novel species text matching (`emb_taxabind_taxctx`).
- **Ensemble Weight**: $w_{\text{tb}} = 1.0$.

### 1.3 `imageomics/bioclip-2` (ViT-L/14 Secondary Image Prototype Backbone)
- **Architecture**: ViT-L/14 dual encoder ($D_{\text{b2}} = 768$).
- **Role**: Supplies a secondary visual prototype representation (`outputs/inat_protos_bioclip2.pt`).
- **Ensemble Weight**: $B2\_W = 2.5$.

---

## 2. Aspect-Shift Matched Adaptation (`ctftshift`)

### 2.1 The Framing Mismatch
Training images use standard crops (aspect ratio $\approx 1.15$), while competition evaluation photos feature elongated fish profiles (aspect ratio $\approx 2.12$).

### 2.2 Shift-Matched Training Recipe (`src/contrastive_ft.py`)
Top 12 residual blocks of BioCLIP ViT-H adapted via LoRA:
```python
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.35, 1.0), ratio=(0.5, 2.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
```

#### LoRA Linear Formulation
For target attention/MLP weights $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:
$$h = W_0 x + \frac{\alpha}{r} (x A^T B^T)$$
- Rank $r = 16$, Alpha $\alpha = 32$ (Scaling $\frac{\alpha}{r} = 2.0$).
- Optimized via InfoNCE Contrastive Loss ($\text{lr} = 5 \times 10^{-4}$, batch size $128$).

### 2.3 Aspect-Squash Extraction (`--squash_tta 1`)
At inference, images are non-isometrically squashed to $224 \times 224$ and averaged with horizontal flips:
$$f(x_i) = \text{normalize}\left(\frac{\text{Encoder}(x_i) + \text{Encoder}(\text{hflip}(x_i))}{2}\right)$$

---

## 3. Model-Derived Soft Novelty Gating

Single-pipeline compliance mandates evaluating all $35,665$ images without split labels. The system computes a joint novelty score $G(x_i)$ for every image:

### 3.1 Component 1: Visual Seen-Prototype Max Score (`img_seenmax`)
$$\text{img\_seenmax}(x_i) = \max_{s \in \mathcal{S}_{\text{seen}}} \mathbf{S}_{\text{seen}}(x_i, s)$$
where $\mathbf{S}_{\text{seen}}(x_i, s)$ represents the fused ensemble confidence score for candidate seen species $s \in \mathcal{S}_{\text{seen}}$. Scores are standardized per-member using z-score normalization ($z_1$).

### 3.2 Component 2: Cross-Split Text Margin (`text_margin`)
$$\text{text\_margin}(x_i) = \max_{s \in \mathcal{S}_{\text{seen}}} (q_i^{\text{ctft}} \cdot T_s^{\text{tax}^T}) - \max_{u \in \mathcal{S}_{\text{unseen}}} (q_i^{\text{ctft}} \cdot T_u^{\text{tax}^T})$$

### 3.3 Unified Gating Metric & Threshold Cutoff ($f = 0.72$)
$$G(x_i) = z_1(\text{img\_seenmax}(x_i)) + 2.0 \cdot z_1(\text{text\_margin}(x_i))$$
$$\theta_{\text{gate}} = \text{TopK}_{k=\lfloor 0.72 \cdot N_{\text{eval}} \rfloor}\Big(\{G(x_i)\}_{i=1}^{N_{\text{eval}}}\Big).\text{min}()$$
$$\text{Route}(x_i) = \begin{cases} \text{Seen Classifier Head} & \text{if } G(x_i) \ge \theta_{\text{gate}} \quad (\text{Top } 72\%) \\ \text{Unseen Classifier Head} & \text{if } G(x_i) < \theta_{\text{gate}} \quad (\text{Bottom } 28\%) \end{cases}$$

---

## 4. Unseen Route Score Fusion & Photo-Bank Maxpooling

For images routed to the unseen candidate space ($U = 11,598$ novel species), `v41` fuses text embeddings with a dual-encoder visual prototype structure:

### 4.1 Fused Text Representation Base
$$S_{\text{text}}(x_i, u) = \text{dbnorm}(Q_i^{\text{ctft}} \cdot T_u^{\text{tax}^T}) + 0.5 \cdot \text{dbnorm}(Q_i^{\text{L}} \cdot T_u^{\text{name}^T}) + 0.75 \cdot \text{dbnorm}(Q_i^{\text{fullft336}} \cdot T_u^{\text{tax}^T}) + 1.0 \cdot \text{dbnorm}(Q_i^{\text{ftshift}} \cdot T_u^{\text{tax}^T}) + 1.0 \cdot \text{dbnorm}(Q_i^{\text{ctft}} \cdot T_u^{\text{taxctx}^T}) + 1.0 \cdot \text{dbnorm}(Q_i^{\text{tb}} \cdot T_u^{\text{tb\_taxctx}^T})$$

### 4.2 iNaturalist Photo-Bank Maxpooling Leg (`TOPM=4`)
For candidate unseen class $u$ with photo bank $\mathbf{G}_u$:
$$S_{\text{raw\_maxpool}}(x_i, u) = \begin{cases} \text{TopK}_{k=\min(4, K_u)}\big(Q_i^{\text{ctft}} \cdot \mathbf{G}_u^T\big).\text{mean}() & \text{if } K_u > 0 \\ -10^4 & \text{if } K_u = 0 \end{cases}$$
$$S_{\text{maxpool\_leg}}(x_i, u) = \text{dbnorm}(S_{\text{raw\_maxpool}}(x_i, u))$$

### 4.3 BioCLIP-2 Prototype Leg (`B2_W=2.5`)
$$S_{\text{raw\_b2}}(x_i, u) = \begin{cases} Q_i^{\text{b2}} \cdot P_{\text{b2}, u}^T & \text{if } \|P_{\text{b2}, u}\| > 0.5 \\ -10^4 & \text{otherwise} \end{cases}$$
$$S_{\text{b2\_leg}}(x_i, u) = \text{dbnorm}(S_{\text{raw\_b2}}(x_i, u))$$

### 4.4 Total Fused Unseen Logit Matrix
$$S_{\text{unseen\_only}}(x_i, u) = S_{\text{text}}(x_i, u) + w_{\text{img}} \cdot S_{\text{maxpool\_leg}}(x_i, u) + 2.5 \cdot S_{\text{b2\_leg}}(x_i, u) \quad (w_{\text{img}} \in \{3.0, 4.0\})$$

---

## 5. Dual-Temperature Normalization (`dbnorm`)

To resolve non-uniform distribution variances across multi-modal backbones:
$$\text{dbnorm}(S; \tau_{\text{col}}=0.05, \tau_{\text{row}}=0.50) = \text{log\_softmax}\left(\frac{S}{0.05}, \text{dim}=0\right) + \text{log\_softmax}\left(\frac{S}{0.50}, \text{dim}=1\right)$$

---

## 6. Transductive Sinkhorn Optimal Transport

To prevent prediction over-concentration, Sinkhorn-Knopp balanced assignment is solved over the routed unseen batch $N_{\text{uns}}$:

$$\mathbf{P}^{(0)} = \text{softmax}\left(\frac{S_{\text{unseen\_only}}}{\tau_{\text{sink}}}, \text{dim}=1\right), \quad \tau_{\text{sink}} = 2.0$$
$$\text{Iterate } K = 50 \text{ steps}: \quad \mathbf{P} \leftarrow \text{normalize\_rows}(\mathbf{P}), \quad \mathbf{P} \leftarrow \text{normalize\_cols}\left(\mathbf{P}, \text{target}=\frac{N_{\text{uns}}}{U}\right)$$
$$\hat{y}_i = \operatorname{argmax}_{u \in \{1, \dots, U\}} P_{i, u}$$

---

## 7. Strict Per-Image Non-Transductive Fallback (`v31`)

For 100% independent per-image inference without batch statistics ([build_v31_strict_singlepipeline.py](file:///home/ubuntu/onet/builders/build_v31_strict_singlepipeline.py)):
$$\hat{y}(x_i) = \begin{cases} \operatorname{argmax}_{s \in \mathcal{S}_{\text{seen}}} S_{\text{seen}}(x_i, s) & \text{if } G(x_i; \text{frozen\_constants}) \ge -2.0423 \\ \operatorname{argmax}_{u \in \mathcal{S}_{\text{unseen}}} S_{\text{unseen}}(x_i, u) & \text{if } G(x_i; \text{frozen\_constants}) < -2.0423 \end{cases}$$
Verified $0 / 35,665$ prediction differences vs. batch-computed builds.
