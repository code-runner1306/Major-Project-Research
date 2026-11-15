# Published-recipe CPU implementation checks

No GPU run or learnability claim. Both MQAR and overwrite completed paired CPU
workflow checks: a two-step timing window followed by a guarded two-epoch
(eight-step) development smoke for scalar and channel-wise GDN. Tiny CPU settings
use32 training examples,vocab128,length32,width16,effective batch8; they do not
represent the100k-example/vocab8192 production benchmark. Both paired manifests
match exactly, source audits pass, actual train/development content overlap is0,
and no confirmation data was generated. Checkpoints stayed in the local scratch
workspace and are excluded here.

57 targeted tests passed. Coverage includes byte-for-byte comparison with the
original MQAR generator at vocab256 and8192; rectangular state forward and all
input gradients; optimizer trajectories; actual scalar/channel common weights;
causality; scored-only CE and every parameter gradient; batched retrieval metrics
against the existing evaluator and explicit erased-answer inputs; and budget
rejection for over-limit/nonfinite/stale evidence. The FLA L2 normalization uses
sum of squares plus1e-6; a zero-vector derivative test guards against accidentally
substituting clamp-based normalization. This port has not been compared against
an actual FLA CUDA kernel, which remains GPU qualification work.

The source-pinned synthetic benchmark is a different setup from the earlier
six-layer/width160 study. Actual standard-size parameter counts are1,585,416
scalar,1,651,456 channel-wise and1,190,144 attention control. Report these counts,
not a nominal10M label. The official-core operations, value expansion, gate
parameterization and initialization differ from the older simplified model;
old checkpoints and learning curves cannot be combined as one experiment.

The copied historical source, its explicit dependency adapters and original
Apache license are in third_party/zoology_iclr24. Pinned MIT-licensed official
GDN/reference-normalization snapshots and attribution are in
third_party/fla_reference. Research notes record paper/code disagreements and
which parts of the new benchmark are our adaptations.

Next required evidence: GPU numerical/model qualification, end-to-end timing
(including setup/validation/checkpoint/publication), measured2/3/4-worker
throughput, common-budget development learning, untouched confirmation, then
calibrated-condition qualification and the final multi-seed protocol lock.
Kaggle access remains blocked by browser control; these CPU passes do not
satisfy those GPU/research requirements.
