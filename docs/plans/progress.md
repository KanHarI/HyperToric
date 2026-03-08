# HyperToric Implementation Progress

## Overview

| Chapter | Status | Branch | Dependencies |
|---------|--------|--------|--------------|
| [WC-1: Project Setup + CI](./wc-01-project-setup-ci.md) | [x] Complete | `wc-1-project-setup-ci` | — |
| [WC-2: Config + Topology](./wc-02-config-topology.md) | [ ] Not Started | `wc-2-config-topology` | WC-1 |
| [WC-3: Taichi Fields](./wc-03-fields.md) | [ ] Not Started | `wc-3-fields` | WC-1, WC-2 |
| [WC-4: Neuron Kernel](./wc-04-neuron-kernel.md) | [ ] Not Started | `wc-4-neuron-kernel` | WC-1, WC-2 |
| [WC-5: Spike Propagation](./wc-05-propagation.md) | [ ] Not Started | `wc-5-propagation` | WC-3, WC-4 |
| [WC-6: STDP Kernels](./wc-06-stdp.md) | [ ] Not Started | `wc-6-stdp` | WC-3, WC-4 |
| [WC-7: Structural Plasticity](./wc-07-plasticity.md) | [ ] Not Started | `wc-7-plasticity` | WC-5, WC-6 |
| [WC-8: Simulator Orchestration](./wc-08-simulator.md) | [ ] Not Started | `wc-8-simulator` | WC-5, WC-6 |
| [WC-9: I/O Manager](./wc-09-io.md) | [ ] Not Started | `wc-9-io` | WC-7, WC-8 |
| [WC-10: Task + Training Loop](./wc-10-task.md) | [ ] Not Started | `wc-10-task` | WC-7, WC-8 |
| [WC-11: Integration Tests](./wc-11-integration.md) | [ ] Not Started | `wc-11-integration` | WC-9, WC-10 |

## Dependency Graph

```
WC-1  ─────────────────────────────  FIRST (gates everything)
  │
  ▼
WC-2  ─────────────────────────────  Config + Topology
  │
  ├───────────┬────────────────────  Parallelizable
  ▼           ▼
WC-3        WC-4                     Fields / Neuron Kernel
  │           │
  ├───────────┤
  ├───────────┬────────────────────  Parallelizable
  ▼           ▼
WC-5        WC-6                     Propagation / STDP
  │           │
  ├───────────┤
  ├───────────┬────────────────────  Parallelizable
  ▼           ▼
WC-7        WC-8                     Plasticity / Simulator
  │           │
  ├───────────┤
  ├───────────┬────────────────────  Parallelizable
  ▼           ▼
WC-9        WC-10                    I/O / Task
  │           │
  ├───────────┤
  ▼
WC-11  ────────────────────────────  Integration Tests
```

## Execution Groups

| Group | Chapters | Can start after |
|-------|----------|-----------------|
| 1 | WC-1 | — |
| 2 | WC-2 | Group 1 |
| 3 | WC-3, WC-4 | Group 2 |
| 4 | WC-5, WC-6 | Group 3 |
| 5 | WC-7, WC-8 | Group 4 |
| 6 | WC-9, WC-10 | Group 5 |
| 7 | WC-11 | Group 6 |

## Verification Checklist (run after each chapter)

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest -x -v
```
