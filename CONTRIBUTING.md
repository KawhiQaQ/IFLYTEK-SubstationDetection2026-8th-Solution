# Contributing

Bug reports and reproducibility improvements are welcome.

1. Do not commit competition data, model weights, credentials, or leaderboard-only labels.
2. Keep the released inference path numerically unchanged unless a pull request explicitly documents and tests the change.
3. Run `python -m unittest discover -s tests -v` and `python -m compileall -q scripts training src` before opening a pull request.
4. New experiments should report the data split, checkpoint provenance, mAP@[0.5:0.95], and whether any test data were read.
