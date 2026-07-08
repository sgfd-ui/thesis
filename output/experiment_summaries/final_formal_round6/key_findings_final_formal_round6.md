# Final Formal Round6 Key Findings

- Main input rows after success filter: `403`.
- Complete main-method points: `46`.
- E4 supplement rows: `15` across `3` cases and `5` methods.

## Overall speedups
- Baseline/Ours `full_skew_join`: geomean `1.475x`, wins for Ours `46/46`.
- Baseline/Ours `amjoin_style`: geomean `1.484x`, wins for Ours `46/46`.
- Baseline/Ours `rdma_onesize`: geomean `1.208x`, wins for Ours `46/46`.
- Baseline/Ours `topology_aware_parallel_join`: geomean `0.729x`, wins for Ours `1/46`.
- Baseline/Ours `aqe_join_reselection`: geomean `0.855x`, wins for Ours `2/46`.

## E1 controlled-work skew
- Controlled speedup vs `full_skew_join`: `1.439x`.
- Controlled speedup vs `amjoin_style`: `1.438x`.
- Controlled speedup vs `rdma_onesize`: `1.211x`.
- Controlled speedup vs `topology_aware_parallel_join`: `0.706x`.
- Controlled speedup vs `aqe_join_reselection`: `0.823x`.

## E3 618 scale and parallelism
- Fig. 8 uses the 20260618 E3 scale/parallelism summary.
- Missing `full_skew_join` values are imputed as `0.97 * amjoin_style` for Fig. 8 only.
- At `128x`: Ours `110.18s`, AMJoin-style `309.15s`, RDMA-OneSize `253.93s`, AQE-style `173.12s`, Topology-aware `154.69s`, imputed Full-SkewJoin `299.88s`.
- Reported scale/parallelism speedup: vs `amjoin_style` `1.77x`; vs `rdma_onesize` `1.44x`.

## E4 supplement
- `ours_kcheck_only`: recall@h `0.865`, precision@h `0.865`, mean time `44.939s`.
- `ours_residual_candidate_only`: recall@h `0.625`, precision@h `0.625`, mean time `45.889s`.
- `ours_no_boundary_lookup`: recall@h `0.948`, precision@h `0.948`, mean time `47.224s`.
- `ours`: recall@h `1.000`, precision@h `1.000`, mean time `46.990s`.
- `oracle_full_rebuild_exact`: recall@h `1.000`, precision@h `1.000`, mean time `41.250s`.
