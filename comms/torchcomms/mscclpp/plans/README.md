# MSCCL++ Execution Plans

This directory holds execution plan JSON files used by `TorchCommMSCCLPP`.

## How plans are loaded

`init()` reads `MSCCLPP_PLAN_DIR` (env var) or the `torchcomm::mscclpp::plan_dir`
hint to locate this directory. Every `*.json` file found is loaded into the plan
cache keyed by its stem (filename without extension). `plan_manifest.json` is
informational only — it documents the expected files and naming convention.

## Obtaining plan files

Plan JSON files are **not bundled** here. Obtain them from the MSCCL++ repository:

```bash
git clone https://github.com/microsoft/mscclpp
ls mscclpp/python/test/execution-files/
```

Copy the relevant files into this directory (or any directory pointed to by
`MSCCLPP_PLAN_DIR`). At minimum, `allreduce_sm_packet.json` and
`allreduce_sm.json` are needed for the `all_reduce` collective.

## Naming convention

`selectPlan()` applies the following auto-selection logic:

| Message size | Key tried              | Fallback   |
|---|---|---|
| ≤ 1 MB       | `<collective>_sm_packet` | `<collective>` |
| > 1 MB       | `<collective>_sm`        | `<collective>` |

Override with the `torchcomm::mscclpp::plan` hint to force a specific plan:

```python
comm.all_reduce(tensor, options={"hints": {"torchcomm::mscclpp::plan": "allreduce_sm"}})
```
