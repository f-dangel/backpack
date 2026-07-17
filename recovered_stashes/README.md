# Recovered stashes (2023 LIFEBOOK backup, rescued 2026-07-17)

Two `git stash` entries from the original kfac-ConvNd working copy, preserved
verbatim. They do not apply cleanly to this branch tip (different bases);
use `git apply -3 <patch>` and resolve, or cherry-pick hunks manually.

- `stash0_on_e1ba4e43_typo_fix.patch` — taken at e1ba4e43: ConvTranspose
  wiring in the HBP extension `__init__`, docs-coverage entries, test glue.
- `stash1_on_14f61c08_kfac_single_output_test.patch` — taken at 14f61c08:
  KFAC/KFLR single-output convolution test work + conv2d HBP edits.
