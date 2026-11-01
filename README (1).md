# 🛡️ IdentityGuard — Multimodal Fake Account Detection

IdentityGuard decides whether a social-media account is **suspect** by combining two signals that are usually studied separately:

1. **The profile** (username, bio, followers, posts, …) → is the account behaving like a bot?
2. **The profile picture** → is the face real, or AI-generated (deepfake)?

A fusion Transformer reads both signals at once. On the held-out test set it reaches **93–94 % accuracy**, versus **~84 %** for either modality alone, and it holds up much better on data sources it has never seen.

> 🎓 Academic project — Master's in Big Data & Data Science, Faculté des Sciences Ben M'Sick, Université Hassan II de Casablanca.
> Topic: *an integrated approach to profile validation and deepfake detection for identity authentication on social networks.*

---

## Table of contents

- [How it works](#how-it-works)
- [Data](#data)
- [Results](#results)
- [Demo app](#demo-app)
- [Getting started](#getting-started)
- [Repository structure](#repository-structure)
- [Limitations](#limitations)
- [References and dataset credits](#references-and-dataset-credits)
- [Author](#author)

---

## How it works

```
 Profile (20 features) ──► FT-Transformer ─┐
                                           ├─► Fusion Transformer ──► suspect / genuine
 Face image ──► ViT-Base (frozen) ──► MLP ─┘
```

| Step | What happens |
|------|--------------|
| **Profile features** | 20 engineered features: digits in the username, display name vs. username, bio length, follower/following ratio, profile completeness, default avatar/theme, verified, … Counts are `log1p`-transformed and standardised (scaler fitted on train only). |
| **Image features** | Face images go through the frozen `google/vit-base-patch16-224-in21k`. The `[CLS]` token gives a 768-dimensional embedding. |
| **Profile encoder** | An FT-Transformer: one learned embedding per feature plus a `[CLS]` token, 2 layers, `d = 64`, 4 heads. |
| **Fusion** | The profile and image tokens receive a learned *modality-type* embedding and are fused by a 2-layer Transformer with its own `[CLS]` token. |
| **Training tricks** | Auxiliary per-modality losses (weight 0.5) and **modality dropout** (30 %), so the model keeps working when one signal is weak or missing. AdamW, early stopping on validation F1, class-weighted loss. |
| **Label** | An account is **suspect** if its profile is fake **or** its image is fake. |

## Data

| Source | Role | Content |
|--------|------|---------|
| **Cresci-2015** (E13, TFP = genuine · FSF, INT, TWT = fake followers) | Twitter profiles | 5,301 accounts |
| **Cresci-stock-2018** (`TWBOT`) | Twitter profiles, added later for more diversity | 12,000 accounts sampled, balanced 50/50 |
| **[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)** | Profile pictures (real vs. StyleGAN faces) | train / valid / test |
| **Instagram Fake Spammer Genuine Accounts** | Cross-platform exploration only (EDA, feature comparison) | — |

After de-duplication: **17,301 accounts**, split **70 / 15 / 15** (stratified on label *and* source, seed 42) → 12,110 train / 2,595 val / 2,596 test. Each image is used **once**.

**Pairing.** No public dataset contains both a profile and its picture, so each profile is **paired with a face image** drawn from the folder of the matching split. To avoid a model that simply "copies" the profile label, **15 % of pairs are deliberately crossed** (e.g. a bot profile with a real face). This is what makes the two modalities genuinely complementary.

## Results

All numbers are on the **test set**, mean over 3 seeds. Suspect = positive class.

### 1. Fusion beats each modality alone

| Model | Accuracy | Precision | Recall | F1 |
|-------|:--------:|:---------:|:------:|:--:|
| Profile only | 0.841 | 0.928 | 0.805 | 0.862 |
| Image only | 0.840 | 0.904 | 0.828 | 0.865 |
| **Fusion** | **0.920** | **0.947** | **0.923** | **0.935** |

Classical baselines on the profile alone (Random Forest, trained on the suspect label): 0.845 accuracy / 0.868 F1.

### 2. Fusion variants

| Variant | Accuracy | F1 |
|---------|:--------:|:--:|
| Fusion (base) | 0.924 | 0.938 |
| + auxiliary losses | 0.931 | 0.943 |
| + auxiliary losses + modality dropout | 0.935 | 0.947 |
| Concatenation + MLP | 0.936 | 0.947 |
| Transformer fusion | 0.934 | 0.946 |
| Transformer fusion + account age features | 0.938 | 0.949 |

Concatenation and Transformer fusion are statistically close on in-distribution data; the Transformer's benefit shows up under distribution shift (below). Account-age features give a small gain but are **excluded from the main model** because they reflect how the datasets were collected (accounts created in batches), not account behaviour.

### 3. Generalisation to unseen sources

The real test of a bot detector: train **without** some data sources, then test on exactly those sources.

| Held-out sources | Profile only (acc / F1) | Fusion Transformer (acc / F1) |
|------------------|:-----------------------:|:-----------------------------:|
| E13 + INT | 0.878 / 0.883 | 0.923 / 0.931 |
| TFP + FSF | 0.926 / 0.952 | 0.958 / 0.973 |
| E13 + TWT | 0.741 / 0.641 | 0.886 / 0.866 |
| TWBOT | 0.471 / 0.248 | 0.764 / 0.768 |
| **Average** | **0.754 / 0.681** | **0.883 / 0.885** |

The profile-only model collapses on new sources (recall 0.15 on `TWBOT`), whereas adding the image keeps performance usable. This confirms the **collection bias** found during the exploratory analysis: profile features alone partly learn *which dataset* an account came from.

### 4. Where the model still struggles

On the "conflicting" pairs — genuine profile with a fake image — accuracy is only ~0.75–0.80 (≈ 0.83 for fake profile + real image). The fusion model arbitrates between contradicting signals but does not resolve every case.

## Demo app

`IdentityGuard v2` is a **Gradio** interface (dark theme, Plotly charts). It loads the trained checkpoint (`multimodal_transformer.pth`) and **does not retrain** anything.

- **Analysis** — fill in a profile form + upload a face photo → suspect probability (gauge), per-modality comparison, feature sensitivity, z-scores
- **Detailed results** — breakdown of the decision
- **About the model** — architecture and data summary
- **History** — previous analyses, exportable to CSV
- **Diagnostics** — live metrics on the test pairs when they are available

> 📸 *Add a screenshot here:* `![IdentityGuard demo](images/demo.png)`

## Getting started

The project was developed and run on **Kaggle** (GPU notebook). The notebook's paths still point to `/kaggle/input/...`; to run it elsewhere, change those paths.

### Option A — Kaggle (easiest)

1. Create a Kaggle notebook with a **GPU** and import `notebook.ipynb`.
2. Add these inputs: Cresci-2015 profiles, Cresci-stock-2018 `users.csv`, [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces), Instagram Fake Spammer Genuine Accounts.
3. Run the cells in order: EDA → preprocessing → pairing → baselines → ViT embeddings → fusion experiments → generalisation test.
4. For the demo, export the trained model as a Kaggle dataset containing `multimodal_transformer.pth`, add it to a fresh notebook, and run the last cell.

### Option B — Local

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

A CUDA GPU is strongly recommended for the ViT embedding step; the full fusion experiments take roughly 3–6 minutes each on a Kaggle GPU.

## Repository structure

```
.
├── notebook.ipynb        # EDA, preprocessing, pairing, models, experiments, Gradio demo
├── requirements.txt
├── README.md
└── images/               # screenshots / figures for this README
```

## Limitations

Being upfront about what this project does **not** prove:

- **Synthetic pairing.** Profiles and faces come from different datasets and are paired artificially. Results show that the *fusion method works*, not that it performs equally on real accounts whose picture and behaviour are genuinely linked.
- **Label rule.** "Suspect = fake profile OR fake image" is a design choice; other rules would change the task.
- **Platform and era.** The profiles are Twitter data from 2015–2018; account features and bot behaviour have evolved. The Instagram data was only used for exploration.
- **Collection bias.** Some sources contain accounts created in batches, which inflates in-distribution scores (hence the held-out-source evaluation).
- **Faces only for one generator family.** The fake faces are StyleGAN-generated; modern diffusion-based deepfakes were not evaluated.
- **Not a production system.** It must not be used to accuse or sanction real people.

## References and dataset credits

- Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., Tesconi, M. — *Fame for sale: efficient detection of fake Twitter followers* (2015) and the Cresci-stock-2018 dataset.
- Gorishniy, Y. et al. — *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer), 2021.
- Dosovitskiy, A. et al. — *An Image is Worth 16x16 Words* (ViT), 2021.
- 140k Real and Fake Faces — Kaggle dataset by *xhlulu*.
- Instagram Fake Spammer Genuine Accounts — Kaggle dataset by *free4ever1*.

Please check each dataset's own licence before redistributing any data. **This repository does not include the datasets.**

## Author

**Aya Moujoud** — Master's student in Big Data & Data Science, Université Hassan II de Casablanca (FSBM).

---

⭐ If you find this project useful, feel free to star the repository.
