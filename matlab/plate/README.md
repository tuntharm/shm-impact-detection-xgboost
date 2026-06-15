# Plate Feature Extraction

Primary entry point: [`datagen_plate.m`](datagen_plate.m)

The plate scripts are preserved as the flat-panel signal-processing stage from the project. They read raw LabVIEW traces, extract the same core feature families, and map impact locations to planar `Loc_X` and `Loc_Y` targets.

The current public Python workflow focuses on the cylindrical tank schema, but the plate scripts document the earlier experimental pipeline and can be adapted if a plate-specific training entrypoint is reintroduced.
