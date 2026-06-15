# Collaboration and Pilot Framing

This project is best read as an SHM research prototype and technical evidence pack. It is suitable for conversations about impact localisation, passive PZT sensing, waveform feature extraction, and machine-learning pipelines for composite structures.

## Potential Evaluation Scenarios

- Lab-scale replication on a new panel or tank coupon.
- Sensor layout study for curved composite structures.
- Feature extraction and ToA benchmarking for guided-wave impact signals.
- Comparison between engineered-feature ML and raw-signal neural baselines.
- Prototype decision support for post-impact inspection triage.

## What a Pilot Would Need

A serious pilot should define:

- target structure geometry and material,
- sensor count, placement, bonding, and acquisition hardware,
- sampling rate, filter settings, and force/impact metadata,
- impact energy range and head/material classes,
- ground-truth location measurement procedure,
- grouped validation protocol before operational claims,
- acceptance criteria for localisation error and classification accuracy,
- safety and maintenance workflow boundaries.

## What This Repository Already Provides

- A cleaned Python training/inference workflow.
- MATLAB feature-extraction lineage for tank and plate experiments.
- Report-backed baseline metrics and model comparison.
- Compact processed sample data with schema and checksums.
- Public commands for training, inference, row-shuffled validation, and grouped validation.
- Limitations and data-access notes for due diligence.

## What Is Not Yet Claimed

The repository does not claim certified inspection readiness, transfer to arbitrary structures, damage severity classification, or anonymous full-data reproducibility. Those would require additional data release, grouped validation, and deployment-specific verification.
