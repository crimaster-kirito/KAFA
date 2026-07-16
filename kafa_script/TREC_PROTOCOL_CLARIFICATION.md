# TREC protocol clarification

The frozen TREC protocol and raw records include
`teacher_temperature: 0.1` in their argument ledger.  The generic
classification runner records this CLI field but does not consume it when
forming client probabilities or the uniform server target.  Client
probabilities are exactly `softmax(-candidate_loss)` and are verified against
the stored probability payloads.

The field is therefore inert metadata, not an applied scientific
hyperparameter.  It is retained in the immutable protocol and records rather
than silently deleting it after official evaluation.  No reported TREC number
depends on a temperature transform.
