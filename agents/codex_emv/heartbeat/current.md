# Current Heartbeat

- Current focus: Muon DIS root-rag workflow tracing.
- Current accepted vertical slice:
  - FairShip preflight automation runs via `scripts/emv_preflight.py`.
  - Muon DIS vertical slice runs end-to-end via `scripts/run_muon_dis_vertical_slice.py`.
  - Parsed text-wrapper hits are converted into qrel candidates via `scripts/run_muon_dis_qrel_review.py`.
  - Project status is reported via `scripts/emv_status.py` (authoritative command).
  - Reviewer-facing qrel table can be generated via `scripts/print_muon_dis_review_sheet.py`.
  - Confirmed qrels remain untouched; candidate qrels remain review-required.
- Current next operational need:
  - manual review of `benchmarks/muon_dis/qrels_review_decisions.yaml`.
  - use review sheet output for targeted manual decisions before promotion.
  - run guarded promotion via `scripts/promote_muon_dis_qrels.py` only after explicit `APPROVED` decisions are written.
  - keep V0 benchmark freeze blocked until enough approved decisions are promoted into `qrels.yaml`.
  - keep wiki claim promotion blocked until review-backed qrels exist.
