# Tank Feature Extraction

Primary entry point: [`datagen.m`](datagen.m)

This stage converts raw tank impact traces into `Features_*.csv` files for the Python models.

## Processing Steps

1. Read LabVIEW time, PZT sensor, and force channels.
2. Low-pass filter each sensor signal.
3. Estimate time of arrival using the vectorised AIC method.
4. Extract signed dominant amplitude.
5. Compute signal energy, using Hilbert-envelope energy for soft impact heads.
6. Convert force-channel voltage into Newtons.
7. Map discrete location IDs to cylindrical `theta` and `z`.
8. Write processed CSVs with one row per impact event.

See [`../../docs/methodology/feature-extraction.md`](../../docs/methodology/feature-extraction.md) for feature definitions and geometry assumptions.
