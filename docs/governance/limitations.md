# Limitations and Governance

This repository is a public technical portfolio and research prototype. It is not a certified inspection system.

## Validity Boundaries

The report-backed results apply to the project dataset, sensor layout, tank geometry, material system, impact setup, and feature-extraction choices used during the FYP. They should not be assumed to transfer directly to:

- different tank radii, materials, wall thicknesses, or boundary conditions,
- pressurised or in-service structures without further validation,
- different PZT placement or sensor count,
- impact energies and head geometries outside the tested cases,
- noisy industrial environments without recalibration,
- safety-critical decisions without independent verification.

## Data Governance

The repository commits compact processed feature CSVs for inspection and sample reproduction. It does not commit raw LabVIEW traces, large experiment folders, generated `.mat` artifacts, or bulky notebook outputs. Full data access remains controlled through Google Drive by the project owner.

## Model Risk

Known risks for future work:

- row-level random splits may overestimate performance when samples are correlated,
- grouped validation by location, case, or impact condition is needed for stronger generalisation claims,
- feature extraction depends on filtering, ToA detection, and calibration constants,
- force conversion and probe scaling should be revalidated for new hardware,
- classification is currently hard/soft impact type rather than damage severity.

## Licensing

No open-source license has been selected. Until a license is added, treat the repository as view-only for evaluation, citation, and discussion. Reuse, redistribution, modification, or commercial application requires explicit permission from the owner. See [`../../NOTICE.md`](../../NOTICE.md).

## Contact

For research collaboration, consulting discussion, or data-access requests, contact the repository owner through GitHub.
