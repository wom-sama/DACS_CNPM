# TRKH Post-CAP Successor No-Run Screen

Date: 2026-07-25

State: complete literature/mechanism screen; no candidate implementation,
training, metric, validation/test access, or model promotion

## Purpose

This screen asks whether a neural-computation family outside the already tested
CNN/Transformer hybrids justifies another TRKH experiment after the failed CAP
A0. It is intentionally stricter than a catalog search. A candidate must:

1. be equation-distinct from closed context, graph, local-mixer, multiplicative,
   morphology, patch-prototype, density, and post-hoc decision routes;
2. add sample-conditional evidence that can distinguish true class 1 from the
   restricted rivals 0, 2, and 4 instead of merely shifting the class-1 prior;
3. support scratch training and a useful information gate within 30 epochs;
4. preserve source-disjoint train-only fitting and exact replay;
5. have a plausible standard-operator deployment path for matched batch-1
   latency, p95 latency, throughput, VRAM, ONNX parity, and TensorRT
   feasibility.

The current keeper, current-best command, and command-history SHA-256 values
remain:

- keeper: `1f49d577240c69dc63c30af70db52ec2aa9da65a17aef1c4b1c09ece6c482677`;
- current-best command:
  `36b9aa1a21b765829acf4c8321be147bd76297de4ccdb8a40e6dee8e37940faf`;
- command history:
  `39bd2879ce66fddf36a953021ea1e40f8d9de6cb4334b9b825011b2b8dc98f53`.

## Candidate Screen

| Family | Primary-source mechanism | TRKH overlap or missing evidence | Deployment concern | Decision |
|---|---|---|---|---|
| Self-organized operational neural networks | Self-ONN learns connection-wise nodal transformations through a Taylor-series polynomial rather than using one fixed convolutional operator. The principal 2D paper evaluates severe image restoration, not fine-grained RGB classification. | The polynomial interaction is adjacent to the already failed MogaNet/StarNet and screened HorNet higher-order or multiplicative routes. It supplies no class-conditional lesion target and does not resolve the observed true-positive versus false-positive tradeoff. | Polynomial order multiplies feature transforms, activation traffic, and export graph size. No matched ONNX/TensorRT classification evidence exists for this host. | No run. Reopen only with scratch image-classification evidence and a surface-selective causal control that is not another generic higher-order stem. |
| Learnable morphological neurons | The BMVC block combines learned dilation and erosion. Its theory and experiments deliberately use 1D operators over the whole input; the paper also notes sparse gradients and slow training for large morphological networks. A separate 2D MorphoN paper targets image de-raining. | Fixed multiscale surface morphology already carried class information but failed to separate true class 1 from restricted false positives. A learned max/min layer is more flexible, but the cited work does not establish that it creates the missing two-sided class-conditional signal on natural RGB classification. | Exact learned 2D max-plus/min-plus neighborhoods require shift/unfold-style expansion and reductions unless reduced to ordinary fixed pooling. A matched standard-op TensorRT path and p95 budget are not demonstrated. | No run. Do not spend GPU on a morphology stem without a prospectively locked CPU operator/export microbenchmark and a source-held frozen-feature selectivity gate. |
| PointNet or Deep Sets over surface tokens | PointNet consumes unordered 3D point sets and obtains permutation invariance through shared point functions and symmetric aggregation. | TRKH has a regular 2D surface grid rather than a 3D point cloud. Removing order from image tokens would reduce the proposal to another patch-set/MIL/prototype aggregator unless a genuinely new input or supervision is supplied. Those local families already failed to create reliable class-1 evidence. | Symmetric pooling itself is deployable, but it does not justify the accuracy experiment; neighborhood variants would reintroduce graph/kNN costs already screened locally. | No run on existing image tokens. |
| DeepEMD or other optimal-transport patch matching | DeepEMD learns relevance through optimal matching flows between dense image regions in a few-shot episodic setting. | The local no-repeat map already screened DeepEMD, DN4, and Sinkhorn as neighbors of failed patch-prototype and part-matching routes. The published task assumes support examples and episodic class matching, not the fixed five-class scratch setting with conflicting surface labels. | Iterative or structured matching adds sample-dependent latency and complicates static deployment relative to the keeper. | No run. |
| Normalizing-flow density head | Normalizing flows provide exact likelihoods through invertible transformations. Primary NeurIPS evidence shows that common flows often learn local pixel correlations and generic transformations rather than semantic dataset identity. | The existing multimodal GMM density head already reduced class-1 true positives and failed the keeper-relative direction gate. A nonlinear flow is a more expensive density neighbor, not new surface supervision, and the cited failure mode is directly adverse to semantic class-1 separation. | Invertible stacks and log-determinant computation add inference work; no mechanism benefit survives the information screen. | No run. |
| Hierarchical local-material recognition | The ICCV method uses graph attention over a physical material taxonomy and a new dataset containing local appearance, depth, and context. | TRKH has five ripeness/damage states, no comparable material taxonomy, and no depth modality. Graph, context, texture aggregation, and class-axis relations have already failed locally. Removing the taxonomy/depth assumptions leaves no distinct mechanism. | The graph head is not the main blocker; the required supervision and modalities are absent. | No run. |

## Primary Sources

- Self-ONN restoration:
  <https://arxiv.org/abs/2008.12894>
- Self-ONN generative neurons:
  <https://www.sciencedirect.com/science/article/pii/S0893608021000782>
- Morphological neurons, BMVC 2022:
  <https://bmvc2022.mpi-inf.mpg.de/0779.pdf>
- 2D morphological image de-raining:
  <https://arxiv.org/abs/1901.02411>
- PointNet, CVPR 2017:
  <https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html>
- DeepEMD, CVPR 2020:
  <https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.html>
- Normalizing-flow semantic-likelihood failure, NeurIPS 2020:
  <https://proceedings.neurips.cc/paper/2020/hash/ecb9fe2fbb99c31f567e9823e884dbec-Abstract.html>
- Hierarchical material recognition, ICCV 2025:
  <https://openaccess.thecvf.com/content/ICCV2025/html/Beveridge_Hierarchical_Material_Recognition_from_Local_Appearance_ICCV_2025_paper.html>

## Decision

None of the six families passes the conjunction. Therefore this screen
authorizes:

- no candidate code;
- no GPU or long CPU training;
- no validation or test access;
- no trainer, dataset, checkpoint, full-train command, or deployment-package
  change.

This is resource-preserving progress, not a model-quality result. The screen
prevents six mechanism-overlapping experiments while retaining the exact
keeper and command boundary.

The next admissible candidate must do more than replace a neuron, mixer,
pooling operator, or density model. Before any image smoke, it must expose
pair-specific local evidence maps or scores for `1-vs-0`, `1-vs-2`, and
`1-vs-4` on bbox-valid fruit surface; fit only source-disjoint train folds;
beat keeper margin and causal derangements on both restricted-FP rejection and
class-1 TP retention; and use a standard-op path that can be measured against
the keeper's complete inference contract.
