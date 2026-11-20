# Gated DeltaNet decay-granularity research

A study of whether giving each key coordinate its own forgetting factor (channel-wise decay) improves associative recall in small Gated DeltaNets, and of what training data teaches those memories.

## Language

### Research organisation

**Experiment**:
One research question with its own result, such as "does decay initialization change the granularity comparison?" or "does the A0–A4 data change remove the shortcut?". An experiment may span several scripts, studies and checkpoints.
_Avoid_: tab, study, run (a run is one trained model)

**Run**:
One trained model, identified by its experiment condition, decay granularity and seed.
_Avoid_: experiment, job

**Arm**:
One training-data condition within the layout-fix experiment (A0 control through A4 distance and rewrites).
_Avoid_: variant, group

### Model

**Decay granularity**:
Whether the decay gate produces one forgetting factor per head (scalar) or one per key coordinate (channel-wise). The only architectural difference between the compared models.
_Avoid_: gating, gate type, gating granularity

**Scalar decay**:
Decay granularity with a single forgetting factor per head and token, applied to the whole recurrent state.
_Avoid_: head-wise gating, DeltaNet

**Channel-wise decay**:
Decay granularity with one forgetting factor per key coordinate and token, applied to the matching key column of the recurrent state.
_Avoid_: KDA-style gating, vector gating, per-channel gating

**Decay gate (α)**:
The data-dependent forgetting factor in (0, 1) that multiplies the recurrent state before each write.
_Avoid_: forget gate, alpha gate

**Write gate (β)**:
The data-dependent strength of the delta-rule correction; one per head in both decay granularities.
_Avoid_: beta gate, learning rate

**Recurrent state**:
The fixed-size value-by-key memory matrix carried from token to token.
_Avoid_: hidden state, cache, memory matrix

**Decay initialization**:
How the decay gate's offset and rate are set before training: default, mean-decay, or joint.
_Avoid_: calibration (except for the mean-decay fitting procedure itself)
