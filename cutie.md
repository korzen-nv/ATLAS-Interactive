# Putting the Object Back into Video Object Segmentation

**Authors:** Ho Kei Cheng$^1$, Seoung Wug Oh$^2$, Brian Price$^2$, Joon-Young Lee$^2$, Alexander Schwing$^1$

$^1$University of Illinois Urbana-Champaign &emsp; $^2$Adobe Research

{hokeikc2, aschwing}@illinois.edu, {seoh, bprice, jolee}@adobe.com

**arXiv:** 2310.12982v2 &emsp; **Date:** 11 Apr 2024

**Code:** [hkchengrex.github.io/Cutie](https://hkchengrex.github.io/Cutie)

---

## Abstract

Cutie is a video object segmentation (VOS) network with object-level memory reading, which puts the object representation from memory back into the video object segmentation result. Recent works on VOS employ bottom-up pixel-level memory reading which struggles due to matching noise, especially in the presence of distractors, resulting in lower performance in more challenging data. In contrast, Cutie performs top-down object-level memory reading by adapting a small set of object queries. Via those, it interacts with the bottom-up pixel features iteratively with a **q**uery-based object **t**ransformer (qt, hence Cutie). The object queries act as a high-level summary of the target object, while high-resolution feature maps are retained for accurate segmentation. Together with foreground-background masked attention, Cutie cleanly separates the semantics of the foreground object from the background. On the challenging MOSE dataset, Cutie improves by 8.7 $\mathcal{J\&F}$ over XMem with a similar running time and improves by 4.2 $\mathcal{J\&F}$ over DeAOT while being three times faster.

---

## 1. Introduction

Video Object Segmentation (VOS), specifically the "semi-supervised" setting, requires tracking and segmenting objects from an open vocabulary specified in a first-frame annotation. VOS methods are broadly applicable in robotics, video editing, reducing costs in data annotation, and can also be combined with Segment Anything Models (SAMs) for universal video segmentation (e.g., Tracking Anything).

Recent VOS approaches employ a memory-based paradigm. A memory representation is computed from past segmented frames (either given as input or segmented by the model), and any new query frame "reads" from this memory to retrieve features for segmentation. Importantly, these approaches mainly use **pixel-level matching** for memory reading, either with one or multiple matching layers, and generate the segmentation bottom-up from the pixel memory readout. Pixel-level matching maps every query pixel independently to a linear combination of memory pixels (e.g., with an attention layer). Consequently, pixel-level matching lacks high-level consistency and is prone to matching noise, especially in the presence of distractors. This leads to lower performance in challenging scenarios with occlusions and frequent distractors. Concretely, the performance of recent approaches is more than 20 points in $\mathcal{J\&F}$ lower when evaluating on the recently proposed challenging MOSE dataset rather than the simpler DAVIS-2017 dataset.

The authors propose **object-level memory reading**, which effectively puts the object from a memory back into the query frame. Inspired by recent query-based object detection/segmentation that represent objects as "object queries," they implement object-level memory reading with an object transformer. This object transformer uses a small set of end-to-end trained object queries to:
1. Iteratively probe and calibrate a feature map (initialized by a pixel-level memory readout)
2. Encode object-level information

This approach simultaneously keeps a high-level/global object query representation and a low-level/high-resolution feature map, enabling bidirectional top-down/bottom-up communication. This communication is parameterized with a sequence of attention layers, including a proposed **foreground-background masked attention**. The masked attention, extended from foreground-only masked attention, lets part of the object queries attend only to the foreground while the remainders attend only to the background -- allowing both global feature interaction and clean separation of foreground/background semantics. Moreover, they introduce a compact **object memory** (in addition to a pixel memory) to summarize the features of target objects, enhancing end-to-end object queries with target-specific features.

### Summary of Contributions

- **Cutie**, which uses high-level top-down queries with pixel-level bottom-up features for robust video object segmentation in challenging scenarios
- **Extended masked attention** to include foreground *and* background for both rich features and a clean semantic separation between the target object and distractors
- A compact **object memory** to summarize object features in the long term, which are retrieved as target-specific object-level representations during querying

---

## 2. Related Works

### Memory-Based VOS

Since semi-supervised VOS involves a directional propagation of information, many existing approaches employ a feature memory representation that stores past features for segmenting future frames. This includes online learning that fine-tunes a network on the first-frame segmentation for every video during inference, though finetuning is slow during test-time. Recurrent approaches are faster but lack context for tracking under occlusion. Recent approaches use more context via pixel-level feature matching and integration, with some exploring the modeling of background features -- either explicitly or implicitly. XMem uses multiple types of memory for better performance and efficiency but still struggles with noise from low-level pixel matching. Cutie develops an object reading mechanism to integrate the pixel features at an object level, attaining much better performance in challenging scenarios.

### Transformers in VOS

Transformer-based approaches have been developed for pixel matching with memory in video object segmentation. However, they compute attention between spatial feature maps (as cross-attention, self-attention, or both), which is computationally expensive with $O(n^4)$ time/space complexity, where $n$ is the image side length. SST proposes sparse attention but performs worse than state-of-the-art methods. AOT approaches use an identity bank for processing multiple objects in a single forward pass to improve efficiency, but are not permutation equivariant with respect to object ID and do not scale well to longer videos. Concurrent approaches use a single vision transformer network to jointly model the reference frames and the query frame without explicit memory reading operations. They attain high accuracy but require large-scale pretraining (e.g., MAE) and have a much lower inference speed (< 4 frames per second). Cutie is carefully designed to *not* compute any (costly) attention between spatial feature maps in the object transformer while facilitating efficient global communication via a small set of object queries -- allowing Cutie to be real-time.

### Object-Level Reasoning

Early VOS algorithms that attempt to reason at the object level use either re-identification or k-means clustering to obtain object features and have a lower performance on standard benchmarks. HODOR, and its follow-up work TarViS, approach VOS with object-level descriptors which allow for greater flexibility (e.g., training on static images only or extending to different video segmentation tasks) but fall short on VOS segmentation accuracy due to under-using high-resolution features. ISVOS proposes to inject features from a pre-trained instance segmentation network (i.e., Mask2Former) into a memory-based VOS method. Cutie has a similar motivation but is crucially different in three ways:
1. Cutie learns object-level information end-to-end, without needing to pre-train on instance segmentation tasks/datasets
2. Cutie allows bi-directional communication between pixel-level features and object-level features for an integrated framework
3. Cutie is a one-stage method that does not perform separate instance segmentation while ISVOS does -- this allows Cutie to run six times (estimated) faster. Moreover, ISVOS does not release code while Cutie is open source

### Automatic Video Segmentation

Recently, video object segmentation methods have been used as an integral component in automatic video segmentation pipelines, such as open-vocabulary/universal video segmentation (e.g., Tracking Anything, DEVA) and unsupervised video segmentation. The robustness and efficiency of Cutie are beneficial for these applications.

---

## 3. Cutie

### 3.1. Overview

Cutie takes a first-frame segmentation of target objects as input and segments subsequent frames sequentially in a streaming fashion.

**Pipeline:**
1. Encode segmented frames (given as input or segmented by the model) into a high-resolution **pixel memory** $F$ and a high-level **object memory** $S$, stored for segmenting future frames
2. To segment a new query frame, retrieve an initial **pixel readout** $R_0$ from the pixel memory using encoded query features (computed via low-level pixel matching, often noisy)
3. Enrich $R_0$ with object-level semantics by augmenting it with information from the object memory $S$ and a set of object queries $X$ through an **object transformer** with $L$ transformer blocks
4. The enriched output $R_L$ (the object readout) is passed to the decoder for generating the final output mask

### 3.2. Object Transformer

#### 3.2.1 Overview

The object transformer takes an initial readout $R_0 \in \mathbb{R}^{HW \times C}$, a set of $N$ end-to-end trained object queries $X \in \mathbb{R}^{N \times C}$, and object memory $S \in \mathbb{R}^{N \times C}$ as input, and integrates them with $L$ transformer blocks. Note $H$ and $W$ are image dimensions after encoding with stride 16. Before the first block, the static object queries are summed with the dynamic object memory for better adaptation:

$$X_0 = X + S$$

Each transformer block bidirectionally allows the object queries $X_{l-1}$ to attend to the readout $R_{l-1}$, and vice versa, producing updated queries $X_l$ and readout $R_l$ as the output of the $l$-th block. The last block's readout, $R_L$, is the final output of the object transformer.

**Within each block:**
1. Compute **masked cross-attention** letting object queries $X_{l-1}$ read from pixel features $R_{l-1}$
2. Pass object queries into standard **self-attention** and **feed-forward layers** for object-level reasoning
3. Update pixel features with a reversed **cross-attention** layer, putting object semantics from $X_l$ back into pixel features $R_{l-1}$
4. Pass pixel features into a **feed-forward network** while skipping self-attention on pixel features

**Key design decisions:**
1. Avoid any direct attention between high-resolution spatial features (e.g., $R$), as they are intensive in both memory and compute. Despite this, spatial features can still interact globally via object queries, making each transformer block efficient and expressive.
2. Object queries restructure pixel features with a residual contribution without discarding the high-resolution pixel features. This avoids irreversible dimensionality reductions (would be over 100x) and keeps those high-resolution features for accurate segmentation.

#### 3.2.2 Foreground-Background Masked Attention

In the (pixel-to-query) cross-attention, the aim is to update the object queries $X_l \in \mathbb{R}^{N \times C}$ by attending over the pixel features $R_l \in \mathbb{R}^{HW \times C}$. Standard cross-attention with the residual path finds:

$$X'_l = A_l V_l + X_l = \text{softmax}(Q_l K_l^T) V_l + X_l \quad (1)$$

where $Q_l$ is a learned linear transformation of $X_l$, and $K_l, V_l$ are learned linear transformations of $R_l$. The rows of the affinity matrix $A_l \in \mathbb{R}^{N \times HW}$ describe the attention of each object query over the entire feature map.

Different object queries have distinctly different attention patterns -- some focus on different foreground parts, some on the background, and some on distractors. These object queries collect information from different regions of interest and integrate them in subsequent self-attention/feed-forward layers. However, the soft nature of attention makes this process noisy and less reliable -- queries that mainly attend to the foreground might have small weights distributed in the background and vice versa.

**Masked attention** is deployed to aid the clean separation of semantics between foreground and background. Different from foreground-only masking, it is helpful to also attend to the background, especially in challenging tracking scenarios with distractors. In practice, the first half of the object queries (i.e., foreground queries) always attend to the foreground and the second half (i.e., background queries) attend to the background. This masking is shared across all attention heads.

**Foreground-background masked cross-attention:**

$$X'_l = \text{softmax}(\mathcal{M}_l + Q_l K_l^T) V_l + X_l \quad (2)$$

where $\mathcal{M}_l \in \{0, -\infty\}^{N \times HW}$ controls the attention masking -- specifically, $\mathcal{M}_l(q, i)$ determines whether the $q$-th query is allowed ($= 0$) or not allowed ($= -\infty$) to attend to the $i$-th pixel.

To compute $\mathcal{M}_l$, a mask prediction is first found at the current layer $M_l$, which is linearly projected from the last pixel feature $R_{l-1}$ and activated with the sigmoid function. Then:

$$\mathcal{M}_l(q, i) = \begin{cases} 0, & \text{if } q \leq N/2 \text{ and } M_l(i) \geq 0.5 \\ 0, & \text{if } q > N/2 \text{ and } M_l(i) < 0.5 \\ -\infty, & \text{otherwise} \end{cases} \quad (3)$$

where the first case is for foreground attention and the second is for background attention.

#### 3.2.3 Positional Embeddings

Since vanilla attention operations are permutation equivariant, positional embeddings are used to provide additional features about the position of each token. Following prior transformer-based vision networks, the positional embedding is added to the query and key features at every attention layer, and not to the value.

**For the object queries**, a positional embedding $P_X \in \mathbb{R}^{N \times C}$ combines an end-to-end learnable embedding $E_X \in \mathbb{R}^{N \times C}$ and the dynamic object memory $S \in \mathbb{R}^{N \times C}$:

$$P_X = E_X + f_{\text{ObjEmbed}}(S) \quad (4)$$

where $f_{\text{ObjEmbed}}$ is a trainable linear projection.

**For the pixel feature**, the positional embedding $P_R \in \mathbb{R}^{HW \times C}$ combines a fixed 2D sinusoidal positional embedding $R_{\sin}$ that encodes absolute pixel coordinates and the initial readout $R_0 \in \mathbb{R}^{HW \times C}$:

$$P_R = R_{\sin} + f_{\text{PixEmbed}}(R_0) \quad (5)$$

where $f_{\text{PixEmbed}}$ is another trainable linear projection. Note that the sinusoidal embedding $R_{\sin}$ operates on normalized coordinates and is scaled accordingly to different image sizes at test time.

### 3.3. Object Memory

The object memory $S \in \mathbb{R}^{N \times C}$ stores a compact set of $N$ vectors which make up a high-level summary of the target object. This object memory is used in the object transformer to provide target-specific features.

At a high level, $S$ is computed by mask-pooling over all encoded object features with $N$ different masks. Concretely, given object features $U \in \mathbb{R}^{THW \times C}$ and $N$ pooling masks $\{W_q \in [0,1]^{THW}, 0 < q \leq N\}$, where $T$ is the number of memory frames, the $q$-th object memory $S_q \in \mathbb{R}^C$ is computed by:

$$S_q = \frac{\sum_{i=1}^{THW} U(i) W_q(i)}{\sum_{i=1}^{THW} W_q(i)} \quad (6)$$

During inference, a classic streaming average algorithm is used such that this operation takes constant time and memory with respect to the video length. Note, an object memory vector $S_q$ would not be modified if the corresponding pooling weights are zero, i.e., $\sum_{i=1}^{THW} W_q(i) = 0$, preventing feature drifting when the corresponding object region is not visible (e.g., occluded).

To find $U$ and $W$ for a memory frame, the corresponding image $I$ and the segmentation mask $M$ are encoded with the mask encoder for memory feature $F \in \mathbb{R}^{THW \times C}$. A 2-layer, $C$-dimensional MLP $f_{\text{ObjFeat}}$ is used to obtain the object feature $U$:

$$U = f_{\text{ObjFeat}}(F) \quad (7)$$

For the $N$ pooling masks $\{W_q \in [0,1]^{THW}, 0 < q \leq N\}$, foreground-background separation is additionally applied and augmented with a fixed 2D sinusoidal positional embedding $R_{\sin}$. The separation allows aggregation of clean semantics during pooling, while the positional embedding enables location-aware pooling.

The $i$-th pixel of the $q$-th pooling mask is computed via:

$$W_q(i) = \begin{cases} 0, & \text{if } q \leq N/2 \text{ and } M(i) < 0.5 \\ 0, & \text{if } q > N/2 \text{ and } M(i) \geq 0.5 \\ \sigma(f_{\text{PoolWeight}}(F(i) + R_{\sin}(i))), & \text{otherwise} \end{cases} \quad (8)$$

where $\sigma$ is the sigmoid function, $f_{\text{PoolWeight}}$ is a 2-layer, $N$-dimensional MLP, and the segmentation mask $M$ is downsampled to match the feature stride of $F$.

### 3.4. Implementation Details

#### 3.4.1 Pixel Memory

The pixel memory design, which provides the pixel feature $R_0$, is derived from XMem working and sensory memory. The pixel memory is composed of an attentional component (with keys $\mathbf{k} \in \mathbb{R}^{THW \times C^k}$ and values $\mathbf{v} \in \mathbb{R}^{THW \times C}$) and a recurrent component (with hidden state $\mathbf{h}^{HW \times C}$). Long-term memory can be optionally included in the attentional component without re-training for better performance on long videos. The keys and values consist of low-level appearance features for matching while the hidden state provides temporally consistent features.

To retrieve a pixel readout $R_0$, the query frame is first encoded to obtain query feature $\mathbf{q}^{HW \times C}$, and the query-to-memory affinity $A^{\text{pix}} \in [0, 1]^{HW \times THW}$ is computed via:

$$A^{\text{pix}}_{ij} = \frac{\exp(d(\mathbf{q}_i, \mathbf{k}_j))}{\sum_m \exp(d(\mathbf{q}_i, \mathbf{k}_m))} \quad (9)$$

where $d(\cdot, \cdot)$ is the anisotropic L2 function which is proportional to the similarity between the two inputs. Finally, $R_0$ is found by combining the attention readout with the hidden state:

$$R_0 = f_{\text{fuse}}(A^{\text{pix}} \mathbf{v} + \mathbf{h}) \quad (10)$$

where $f_{\text{fuse}}$ is a small network consisting of two $C$-dimension convolutional residual blocks with channel attention.

#### 3.4.2 Network Architecture

Two model variants are studied: 'small' and 'base' with different query encoder backbones, otherwise sharing the same configuration: $C = 256$ channels with $L = 3$ object transformer blocks and $N = 16$ object queries.

**ConvNets.** The query encoder and the mask encoder are parameterized with ResNets. Following prior work, the last convolutional stage is discarded and stride 16 features are used. For the query encoder, ResNet-18 is used for the small model and ResNet-50 for the base model. For the mask encoder, ResNet-18 is used. 'Cutie-base' thus shares the same backbone configuration as XMem. Cutie works well with a lighter decoder -- a similar iterative upsampling architecture as in XMem is used but with halved number of channels in all upsampling blocks for better efficiency.

**Feed-Forward Networks (FFN).** Both query FFN and pixel FFN are used in the object transformer block. For the query FFN, a 2-layer MLP with a hidden size of $8C = 2048$. For the pixel FFN, two $3 \times 3$ convolutions with a smaller hidden size of $C = 256$ to reduce computation. Since self-attention on the pixel features is not used, efficient channel attention is used after the second convolution of the pixel FFN. Layer normalizations are applied to the query FFN following prior work and not to the pixel FFN, as no empirical benefits were observed. ReLU is used as the activation function.

#### 3.4.3 Training

**Data.** The network is first pretrained on static images by generating three-frame sequences with synthetic deformation. The main training is then performed on video datasets DAVIS and YouTubeVOS by sampling eight frames. Optionally, training is also done on MOSE (combined with DAVIS and YouTubeVOS), as the training sets of YouTubeVOS and DAVIS have become too easy for the model to learn from (>93% IoU during training). For every setting, one trained model is used and specific datasets are not finetuned for. Additionally a 'MEGA' setting is introduced with BURST and OVIS included in training (+1.6 $\mathcal{J\&F}$ in MOSE).

**Optimization.** The AdamW optimizer with a learning rate of 1e-4, a batch size of 16, and a weight decay of 0.001. Pretraining lasts for 80K iterations with no learning rate decay. Main training lasts for 125K iterations, with the learning rate reduced by 10 times after 100K and 115K iterations. The query encoder has a learning rate multiplier of 0.1 to mitigate overfitting. The global gradient norm is clipped to 3 throughout and stable data augmentation is used. The entire training process takes approximately 30 hours on four A100 GPUs for the small model.

**Losses.** Point supervision is adopted which computes the loss only at $K$ sampled points instead of the whole mask. Importance sampling is used and $K = 8192$ during pretraining and $K = 12544$ during main training. A combined loss function of cross-entropy and soft dice loss with equal weighting is used. In addition to the loss applied to the final segmentation output, auxiliary losses in the same form (scaled by 0.01) are adopted to the intermediate masks $M_l$ in the object transformer.

#### 3.4.4 Inference

A memory frame is encoded for updating the pixel memory and the object memory every $r$-th frame. $r$ defaults to 5. For the keys $\mathbf{k}$ and values $\mathbf{v}$ in the attention component of the pixel memory, features from the first frame are always kept (as it is given by the user) and a First-In-First-Out (FIFO) approach is used for other memory frames to ensure the total number of memory frames $T$ is less than or equal to a pre-defined limit $T_{\max} = 5$. For processing long videos (e.g., BURST or LVOS with over a thousand frames per video), the long-term memory from XMem is used instead of FIFO without re-training. For the pixel memory, top-$k$ filtering is used with $k = 30$. Inference is fully online, can be streamed, and uses a constant amount of compute per frame and memory with respect to the sequence length.

---

## 4. Experiments

Standard metrics are used for evaluation: Jaccard index $\mathcal{J}$, contour accuracy $\mathcal{F}$, and their average $\mathcal{J\&F}$. In YouTubeVOS, $\mathcal{J}$ and $\mathcal{F}$ are computed for "seen" and "unseen" categories separately. $\mathcal{G}$ is the averaged $\mathcal{J\&F}$ for both seen and unseen classes. For BURST, Higher Order Tracking Accuracy (HOTA) is assessed on common and uncommon object classes separately. Unless otherwise specified, inputs are resized such that the shorter edge has no more than 480 pixels and the prediction is rescaled back to the original resolution.

### 4.1. Main Results

#### Table 1: Quantitative comparison on video object segmentation benchmarks

**Trained without MOSE:**

| Method | MOSE $\mathcal{J\&F}$ | MOSE $\mathcal{J}$ | MOSE $\mathcal{F}$ | DAVIS-17 val $\mathcal{J\&F}$ | DAVIS-17 test $\mathcal{J\&F}$ | YT-VOS $\mathcal{G}$ | FPS |
|--------|--------|------|------|----------|----------|------|-----|
| STCN | 52.5 | 48.5 | 56.6 | 85.4 | 76.1 | 82.7 | 13.2 |
| AOT-R50 | 58.4 | 54.3 | 62.6 | 84.9 | 79.6 | 85.3 | 6.4 |
| RDE | 46.8 | 42.4 | 51.3 | 84.2 | 77.4 | 81.9 | 24.4 |
| XMem | 56.3 | 52.1 | 60.6 | 86.2 | 81.0 | 85.5 | 22.6 |
| DeAOT-R50 | 59.0 | 54.6 | 63.4 | 85.2 | 80.7 | 85.6 | 11.7 |
| DEVA | 60.0 | 55.8 | 64.3 | 86.8 | 82.3 | 85.5 | 25.3 |
| **Cutie-small** | **62.2** | **58.2** | **66.2** | 87.2 | 84.1 | **86.2** | **45.5** |
| **Cutie-base** | **64.0** | **60.0** | **67.9** | **88.8** | **84.2** | 86.1 | 36.4 |

**Trained with MOSE:**

| Method | MOSE $\mathcal{J\&F}$ | DAVIS-17 val $\mathcal{J\&F}$ | DAVIS-17 test $\mathcal{J\&F}$ | YT-VOS $\mathcal{G}$ | FPS |
|--------|--------|----------|----------|------|-----|
| XMem | 59.6 | 86.0 | 79.6 | 85.6 | 22.6 |
| DeAOT-R50 | 64.1 | 86.0 | 82.8 | 85.3 | 11.7 |
| DEVA | 66.0 | 87.0 | 82.6 | 85.4 | 25.3 |
| **Cutie-small** | 67.4 | 86.5 | 83.8 | **86.3** | **45.5** |
| **Cutie-base** | **68.3** | **88.8** | **85.3** | **86.5** | 36.4 |

#### Table 2: BURST dataset (long videos) -- performance and GPU memory usage

| Method | Mem. Type | BURST val All | BURST test All | Mem. |
|--------|-----------|--------|--------|------|
| DeAOT FIFO | w/ MOSE | 51.3 | 53.2 | 10.8G |
| DeAOT INF | w/ MOSE | 56.4 | 57.9 | 34.9G |
| XMem FIFO | w/ MOSE | 52.9 | 55.9 | 3.03G |
| XMem LT | w/ MOSE | 55.1 | 58.2 | 3.34G |
| **Cutie-small FIFO** | w/ MOSE | 56.8 | 61.1 | **1.35G** |
| **Cutie-small LT** | w/ MOSE | 58.3 | 61.6 | 2.28G |
| **Cutie-base LT** | w/ MOSE | **58.4** | **62.6** | 2.36G |

Cutie achieves better results than state-of-the-art methods, especially on the challenging MOSE dataset, while remaining efficient. Cutie-small with FIFO memory uses only **1.35G** GPU memory.

### 4.2. Ablations

All ablations use the small model variant with MOSE training data.

#### Table 3: Hyperparameter Choices

**Number of transformer blocks ($L$):**

| Setting | $\mathcal{J\&F}$ | FPS |
|---------|------|-----|
| $L = 0$ | 65.2 | 56.6 |
| $L = 1$ | 66.0 | 51.1 |
| **$L = 3$** | **67.4** | **45.5** |
| $L = 5$ | 67.8 | 37.1 |

**Number of object queries ($N$):**

| Setting | $\mathcal{J\&F}$ | FPS |
|---------|------|-----|
| $N = 8$ | 67.6 | 45.5 |
| **$N = 16$** | **67.4** | **45.5** |
| $N = 32$ | 67.2 | 45.5 |

**Memory interval ($r$):**

| Setting | $\mathcal{J\&F}$ | FPS |
|---------|------|-----|
| $r = 3$ | 68.9 | 43.2 |
| **$r = 5$** | **67.4** | **45.5** |
| $r = 7$ | 67.0 | 46.4 |

**Maximum memory frames ($T_{\max}$):**

| Setting | $\mathcal{J\&F}$ | FPS |
|---------|------|-----|
| $T_{\max} = 3$ | 66.9 | 48.5 |
| **$T_{\max} = 5$** | **67.4** | **45.5** |
| $T_{\max} = 10$ | 67.6 | 37.4 |

The object transformer blocks effectively suppress noises from distractors and produce more coherent object masks. Cutie is insensitive to the number of object queries -- 8 queries are sufficient to model the foreground/background of a single target object. Cutie benefits from a shorter memory interval and a larger memory bank at the cost of a slower running time.

#### Table 4: Bottom-Up vs. Top-Down Feature

| Setting | $\mathcal{J\&F}$ | FPS |
|---------|------|-----|
| **Both** | **67.3 $\pm$ 0.36** | **45.5** |
| Bottom-up only | 65.0 $\pm$ 0.44 | 56.6 |
| Top-down only | 40.7 $\pm$ 1.62 | 46.9 |

Integrating both features performs the best.

#### Table 5: Dynamic object memory ($S$) and static object query ($X$)

| Setting | $\mathcal{J\&F}$ |
|---------|------|
| **With both** | **67.3 $\pm$ 0.36** |
| No object memory ($X$) | 66.9 $\pm$ 0.26 |
| No object query ($S$) | 67.2 $\pm$ 0.10 |

The object query, while standard, is not as useful for Cutie in the presence of the object memory.

#### Table 6: Masked Attention

| Setting | $\mathcal{J\&F}$ | FPS |
|---------|------|-----|
| **f.g.-b.g. masked attn.** | **67.3 $\pm$ 0.36** | **45.5** |
| f.g. masked attn. only | 66.7 $\pm$ 0.21 | 45.5 |
| No masked attn. | 63.8 $\pm$ 1.06 | 46.3 |

Masked attention is crucial for good performance -- using full attention produces confusing signals (especially in cluttered settings), which leads to poor generalization. Using full attention also leads to rather unstable training.

#### Table 7: Positional Embeddings

| Setting | $\mathcal{J\&F}$ |
|---------|------|
| **With both p.e.** | **67.4** |
| Without query p.e. | 66.5 |
| Without pixel p.e. | 66.2 |
| With neither | 66.1 |

Positional embeddings are commonly used and do help.

### 4.3. Limitations

Despite being more robust, Cutie often fails when highly similar objects move in close proximity or occlude each other. This problem is not unique to Cutie. In these cases, neither the pixel memory nor the object memory is able to pick up sufficiently discriminative features for the object transformer to operate on. A potential future work direction is to encode three-dimensional spatial understanding.

---

## 5. Conclusion

Cutie is an end-to-end network with object-level memory reading for robust video object segmentation in challenging scenarios. Cutie efficiently integrates top-down and bottom-up features, achieving new state-of-the-art results in several benchmarks, especially on the challenging MOSE dataset. The hope is to draw more attention to object-centric video object segmentation and to enable more accessible universal video segmentation methods via integration with segment-anything models.

---

## Supplementary Material

### A. Visual Comparisons

Visual comparisons of Cutie with DeAOT-R50 and XMem are available at [youtu.be/LGbJ11GT8Ig](https://youtu.be/LGbJ11GT8Ig). Cutie-base is used and all models are trained with the MOSE dataset. Comparisons are visualized on YouTubeVOS-2019 validation, DAVIS 2017 test-dev, and MOSE validation. The model is more robust to distractors and generates more coherent masks.

### B. Failure Cases

Failure cases are visualized at [youtu.be/PIjXUYRzQ8Q](https://youtu.be/PIjXUYRzQ8Q). Cutie fails in some of the challenging cases where similar objects move in close proximity or occlude each other. This is due to the lack of useful features from the pixel memory and the object memory, as they fail to disambiguate objects that are similar in both appearance and position.

### C. Running Time Analysis

Total running time (s) of each component (tested on a single video with a 2080Ti):

| Component | XMem | Cutie-base | Cutie-small |
|-----------|------|------------|-------------|
| Query encoder | 0.861 | 0.851 | 0.295 |
| Mask encoder | 0.143 | 0.145 | 0.142 |
| Pixel memory read | 0.758 | 0.514 | 0.514 |
| Object memory read | - | 0.913 | 0.894 |
| Decoding | 2.749 | 0.725 | 0.700 |

The speedup is mostly achieved by using a lighter decoder.

### F. Additional Quantitative Results

#### F.1. Speed-Accuracy Trade-off

"Cutie+" adjusts the following hyperparameters without re-training:
1. Maximum memory frames $T_{\max} = 5 \to T_{\max} = 10$
2. Memory interval $r = 5 \to r = 3$
3. Maximum shorter side resolution during inference $480 \to 720$ pixels

These settings apply to DAVIS and MOSE. For YouTubeVOS, $r = 5$ is kept and the maximum shorter side resolution is set to 600.

### G. Implementation Details

#### G.1. Extension to Multiple Objects

Cutie extends to the multi-object setting by processing objects independently (in parallel as a batch) except for:
1. The interaction at the first convolutional layer of the mask encoder (5-channel input: image + target mask + non-target masks)
2. The interaction at the soft-aggregation layers used to generate segmentation logits (probability distributions sum to one at every pixel)

The method remains real-time when handling a common number of objects (29.9 FPS with 5 objects).

#### G.2. Streaming Average Algorithm for Object Memory

During inference, for the $q$-th object memory at time step $t$, a cumulative memory $\sigma^t_{S_q} \in \mathbb{R}^C$ and a cumulative weight $\sigma^t_{W_q} \in \mathbb{R}$ are tracked. The accumulators are updated and $S_q$ is found via:

$$\sigma^t_{S_q} = \sigma^{t-1}_{S_q} + \sum_{i=1}^{THW} U(i) W_q(i), \quad \sigma^t_{W_q} = \sigma^{t-1}_{W_q} + \sum_{i=1}^{THW} W_q(i), \quad \text{and} \quad S_q = \frac{\sigma^t_{S_q}}{\sigma^t_{W_q}}$$

where $U$ and $W_q$ can be discarded after every time step.

#### G.3. Training Details

- **Pretraining:** Static image pretraining using datasets ECSSD, DUTS, FSS-1000, HRSOD, and BIG. Three-frame synthetic sequences are generated using random affine transformation, thin plate spline transformation, and cropping (crop size $384 \times 384$).
- **Main Training:** Two settings -- "without MOSE" mixes DAVIS-2017 and YouTubeVOS-2019; "with MOSE" adds MOSE. DAVIS is sampled 2x more often. A "seed" frame is randomly selected and seven other frames are chosen from the same video. Curriculum learning schedule for max frame distance $D$ is set to $[5, 10, 15, 5]$ after $[0\%, 10\%, 30\%, 80\%]$ of training iterations.
- **Data Augmentation:** Random horizontal mirroring, random affine transformation, cut-and-paste from another video, random resized crop (scale [0.36, 1.00], crop size 480 x 480), stable data augmentation (same crop and rotation for all frames), random color jittering, and random grayscaling.
- **Point Supervision:** Importance sampling with oversampling ratio 3, sampling 75% of all points from uncertain points and the rest from a uniform distribution.

#### G.4. Decoder Architecture

The decoder follows XMem with a reduced number of channels (128 vs 256 in XMem). The inputs to the decoder are the object readout feature $R_L$ at stride 16 and skip-connections from the query encoder at strides 8 and 4. The skip-connection features are first projected to $C$ dimensions with a $1 \times 1$ convolution. In each upsampling block, the input feature is bilinearly upsampled by 2x, added with the skip-connection features, then processed by a residual block with two $3 \times 3$ convolutions. In the final layer, a $3 \times 3$ convolution predicts single-channel logits for each object. The logits are bilinearly upsampled to the original input resolution. In the multi-object scenario, soft-aggregation is used to merge the object logits.

---

## Key Configuration Summary

| Parameter | Value |
|-----------|-------|
| Channels ($C$) | 256 |
| Transformer blocks ($L$) | 3 |
| Object queries ($N$) | 16 |
| Memory interval ($r$) | 5 |
| Max memory frames ($T_{\max}$) | 5 |
| Query encoder (small) | ResNet-18 |
| Query encoder (base) | ResNet-50 |
| Mask encoder | ResNet-18 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 16 |
| Weight decay | 0.001 |
| Pretraining iterations | 80K |
| Main training iterations | 125K |
| Loss | Cross-entropy + Soft dice |
| Top-k filtering | $k = 30$ |
