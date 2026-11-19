# Independent novelty review: scalar/channel gating, MQAR, and overwrite

Date: 17 November 2026

## Bottom line

The broad claim that channel-wise gating is a new idea, or that it has never
been compared with scalar gating, is no longer defensible. A TMLR manuscript
by Satyajit Das presents a controlled factorial study with scalar versus
channel-wise decay, data-dependent versus data-independent gates, and with or
without the delta rule. Its cells include DeltaNet, KDA, ScalarStaticDelta,
and StaticChannelDelta, with three seeds at 18M and 125M parameters. It also
contains a synthetic associative-recall probe where scalar decay plus the
delta rule succeeds and channel-wise decay variants fail at the tested scale.
See the [TMLR page](https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html),
the [paper text](https://openreview.net/pdf?id=FdXB4u27LD), and the
[released code and results](https://github.com/emailtovamos/delta-rule-linear-attention).

The exact current project is therefore not publishable as “the first scalar
versus channel-wise gating comparison.” Its defensible contribution must be
narrower and explicitly comparative: a small, reproducible Gated DeltaNet
study of how activation-dependent decay, matched initial decay, task type,
and repeated-write robustness interact. Even that claim needs to acknowledge
the close factorial prior work and avoid presenting the result as a new gate
architecture.

## Closest direct overlap: Delta-Rule Writes in Linear Attention

The TMLR study evaluates eight matched attention variants. Its design crosses
granularity (scalar/channel-wise), conditioning (token-dependent/data-
independent), and the delta rule. StaticChannelDelta has a fixed learned
per-channel decay and a delta-rule write; ScalarStaticDelta is its scalar
counterpart. The paper explicitly describes the channel gate as one value per
state dimension, fixed across input positions, and initializes the values on
an exponential timescale spectrum (approximately one to 163 tokens at its
64-channel setting). See Sections 3.2–3.4 of the paper text.

This is stronger overlap than a generic related citation:

* The comparison is factorial and multi-seed, rather than a single baseline.
* It studies the same family-level distinction between a scalar and a
  channel-wise decay gate around a delta-rule memory.
* It includes a synthetic associative-recall probe and analyzes whether
  channel-wise decay fragments key–value bindings.
* It reports that StaticChannelDelta leads the 18M language-model runs, while
  at 125M the ordering against KDA is close and dataset-dependent.

The differences that remain useful for our study are concrete. The TMLR
factorial cells use data-independent static gates or the published KDA/DeltaNet
cells; they do not perform our activation-sample procedure that matches the
initial mean decay of the token-dependent scalar and channel-wise GDN models.
Its associative-recall probe uses one fresh tiny model per variant, a single
query, four disjoint key–value pairs, and distractor distances up to 512. It
does not use our 3-seed current recipe, 512-example validation cadence, or our
answer-free repeated-overwrite metric with unrelated-association retention.
Those differences make a replication/boundary study possible, but they do not
erase the direct prior art.

The TMLR manuscript reports: DeltaNet and ScalarStaticDelta are near-perfect,
while KDA and StaticChannelDelta stay around 27% in its 5.1M one-seed probe;
doubling the budget from 15K to 30K steps changes the channel-wise variants by
only about one percentage point. The authors interpret this as a possible
coherence problem when different dimensions of a binding decay at different
rates. Our small GDN checkpoints show a different learnability boundary on the
current MQAR recipe, so this is best framed as an explicit replication and
task/protocol comparison, not as a discovery that channel-wise gates are
universally beneficial.

## Initialization and calibration overlap

The planned third experiment is also not novel as a general initialization
idea. DAMI (Feng et al., arXiv:2610.41994v2) jointly initializes `A_log` and
`dt_bias` from a small activation sample. It sets geometric head-wise
timescales and a low–high–low layer sensitivity profile, solving the bias and
decay targets layer by layer. The method is explicitly data-dependent and
targets the same GDN decay equation used here. See [DAMI](https://arxiv.org/html/2610.41994v2),
especially Sections 3.2–3.5 and Equations 9–11.

Our earlier calibration changed only `dt_bias` and matched a mean decay target;
that is a simpler calibration diagnostic, not a new initialization family.
The new launcher calls its method a **DAMI-style adaptation** because our
one-head/four-layer model cannot reproduce DAMI’s multi-head spectrum exactly.
The manuscript must say which parts are reproduced, which are adapted, and
which are new controls. It must not claim to introduce joint timescale and
sensitivity initialization.

PoST (Cao, arXiv:2604.07658) imposes geometric decay spectra and
position-adaptive scaling across linear recurrences, including Gated DeltaNet.
StaticChannelDelta also uses an exponential decay spectrum. These works make
“multi-timescale decay improves memory” an established direction rather than
an available novelty claim.

Older foundations matter as well. Chrono initialization for gated RNNs
(Tallec and Ollivier, arXiv:1804.11188) established initializing gate biases
from expected dependency horizons. Fast Weight Programmers / DeltaNet
(Schlag et al., arXiv:2102.11174) established the online delta-rule memory
interpretation and its value for correcting associations.

## MQAR and overwrite overlap

Zoology introduced Multi-Query Associative Recall and showed why fixed-state
gated sequence models can struggle when multiple associations must be recalled.
See [Zoology](https://arxiv.org/abs/2312.04927). MQAR is an established
diagnostic, not a new benchmark in this project.

The closest recent overwrite comparison is KATA (Ghriss and Chakraborty,
arXiv:2607.17419). Its public paper has a dedicated “Repeated-key overwrite”
section: each key is written twice with different values, and evaluation asks
for the latest binding. Training uses up to 32 keys and evaluation extends to
64–384 keys and contexts up to 3072. The paper compares Gated DeltaNet and
other fixed-state methods and reports both MQAR length extrapolation and
repeated-key overwrite. See [KATA](https://arxiv.org/html/2607.17419v1),
Sections 5.1–5.2 and Appendix F.

Our overwrite test is distinguishable in protocol: it uses audited keyed v4
data, a 64-token current recipe, one rewrite in training, and a post-hoc stress
grid including three rewrites, farther queries, more keys, and unrelated-
association retention. The error breakdown separating the initial value,
first rewrite, second rewrite, and “other” answers is a useful mechanistic
extension. But “we test latest-write-wins overwrite in a linear attention
memory” is already claimed by KATA and related delta-rule work.

Gated DeltaNet-2 (Hatamizadeh et al., arXiv:2605.22791) explicitly generalizes
KDA with channel-wise erase and channel-wise write gates, separating the two
roles. See [GDN-2](https://arxiv.org/abs/2605.22791). Muir (Gangavarapu et al.,
arXiv:2611.08376) studies channel-wise gated delta memories, multi-query
recall, overwriting writes, and counterfactual gate interventions at multiple
seeds and scales. See [Muir](https://arxiv.org/html/2611.08376). Our gate
intervention plots should be presented as a small-model, matched-decay
diagnostic rather than a new general methodology.

## What can still be a credible contribution

The defensible angle is an empirical boundary/replication study with a tightly
specified claim, for example:

> In a 1.6M-parameter, one-head GDN adaptation, matching the initial realized
> decay of scalar and channel-wise token-dependent gates does not remove
> seed-dependent differences between MQAR and repeated overwrite. Clean MQAR
> can remain near-perfect while multi-rewrite stress reveals obsolete-value
> errors; we quantify those errors and test whether they are attributable to
> channel heterogeneity, decay input dependence, or initialization.

That claim is potentially useful because it reconciles apparently conflicting
results: the TMLR study’s channel-wise associative-recall failure, our current
recipe’s clean MQAR learnability, and the overwrite stress failures. It is a
careful cross-protocol finding, not a new architecture. To support it, the
paper needs:

1. A prior-art baseline reproducing at least the relevant scalar/channel and
   static/dynamic cells, or a very explicit explanation of why exact
   reproduction is impossible.
2. A prespecified calibration comparison: random/default, mean-decay matching,
   and DAMI-style joint calibration. Do not tune target half-lives after seeing
   stress results.
3. Fresh held-out evaluation for MQAR and overwrite, including repeated writes
   and unrelated retention, with three seeds and confidence intervals or
   per-seed results.
4. A parameter-matched control, because the present channel-wise model has
   approximately 4.17% more trainable parameters.
5. A limitation statement that 1.6M parameters and synthetic tasks establish
   mechanism evidence, not production-scale GDN performance.

If the paper instead claims “channel-wise gating is novel,” “we introduce
calibrated GDN initialization,” or “we are the first to study overwrite in
linear attention,” the novelty claim will likely fail review.

## Search scope and confidence

This review checked the primary arXiv/OpenReview papers and author-released
repositories for GDN, KDA, GDN-2, DAMI, PoST, Muir, KATA, Zoology, and the
closest controlled factorial study, including historical delta-rule and chrono
initialization work. It is not a proof that no unpublished, non-indexed,
industry, or simultaneous paper exists. Search results are current to
17 November 2026. The direct TMLR overlap is high-confidence because the paper
text and code repository expose the exact cells, seeds, equations, and
associative-recall protocol.
