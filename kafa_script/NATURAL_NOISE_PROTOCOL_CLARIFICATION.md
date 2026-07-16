# Natural-noise protocol clarification

This note preserves, rather than rewrites, the frozen Hausa and Yoruba
protocols.  Their statement that clean training labels are used "only for
selection audit" is too narrow.

Before any model training, the runner compares the released weak and clean
training labels to compute one corpus-level disagreement rate.  That scalar is
the oracle Co-Teaching forget rate and therefore affects how many examples are
retained.  Models are still trained on weak labels only: per-example clean
labels do not enter loss computation, peer scoring, ranking, gradient updates,
or evaluation.  They are retained as `kind` metadata solely to audit the clean
fraction of selected examples.  Clean development labels are used for the
development gate; the clean test splits were not evaluated.

Relevant implementation: `kafa_script/run_natural_noise_coteaching_dev.py`
computes the disagreement rate before training and uses it as `forget_rate`;
the clean/weak match indicator is read only when accumulating selection-audit
counts.

This clarification was added after the development results were observed.  It
changes no run, record, gate, or reported number.
