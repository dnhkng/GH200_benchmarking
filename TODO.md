# TODO

These items need sudo or persistent system tuning and were not executed by the benchmark runner.

- Enable GPU persistence and lock GPU clocks for tighter run-to-run variance: `sudo nvidia-smi -pm 1` and `sudo nvidia-smi -lgc <max>`.
- Optional: configure a persistent huge-page pool and rerun STREAM/BabelStream/custom host-memory tests. NVBandwidth `-H` completed without sudo on 2026-04-25, but broader system tuning still requires elevated privileges.

Completed on 2026-04-25:

- CPU performance governor enabled by user with `sudo cpupower frequency-set -g performance`.
- THP `always` enabled by user with `echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled`.
