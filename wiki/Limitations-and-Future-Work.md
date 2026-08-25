# Limitations and Future Work

## Limitations

- Only 500 images and 80 final-test samples.
- Ten identities, far below the original large-scale supervision.
- Closed-set classification with no unknown-person rejection.
- One full-face network instead of multiple face-patch ConvNets.
- Identification rather than standard LFW pair verification.
- One fixed split, so sampling variance is not estimated.
- Frequent LFW identities do not represent real-world demographics.

## Recommended next steps

1. Establish a new protocol with repeated stratified splits and report mean ± standard deviation.
2. Add face-patch branches to test the original multi-region hypothesis.
3. Compare cross-entropy with center, contrastive, triplet, or angular-margin losses.
4. Compare training from scratch with a pretrained face backbone.
5. Move to standard verification metrics such as ROC, TAR at FAR, and pair accuracy.
6. Add calibrated rejection and a true open-set evaluation.

The current final test must not be reused for iterative tuning. Further model changes require a newly declared evaluation protocol.
