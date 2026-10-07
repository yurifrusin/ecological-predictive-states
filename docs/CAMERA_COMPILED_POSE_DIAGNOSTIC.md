# Compiled camera pose diagnostic

`epsbench.diagnostics.corridor_aperture_capture.CameraCompilationDiagnostic` provides a disabled-by-default, privileged source operation for inspecting the first fixed camera pose after MuJoCo compilation and `mj_forward`. It records the complete observed position and orientation, frozen expected values, componentwise exact comparisons, finite hexadecimal float spellings, exact source/configuration binding, and MuJoCo version through the supplied privileged retention sink.

The callable requires a typed permission set containing every modality in the aperture `PRIVILEGED` set, `enabled=True`, an exact `SourceBinding`, and a retention sink. It checks the bound clean source before loading MuJoCo. Its shared setup compiles the fixed XML, checks the nine named geometries and camera, assigns the frozen first-view position, creates `MjData`, and calls `mj_forward`. The diagnostic stops there: it does not create or update a renderer, draw, access raster data, certify labels or frames, or allocate episode identities.

The returned bytes are canonical privileged diagnostics marked `unvalidated_privileged_compiled_camera_pose`. A retention error or malformed/nonfinite operand fails closed. Native capture also retains this record before constructing and validating the frame, so an exact pose rejection retains the operands that caused it. This additional operational record does not alter scientific frame bytes or frozen validation rules.

The result applies only to the newly run compilation diagnostic. It cannot recover operands omitted by an earlier attempt, establish a draw-path result, certify a frame, or change the frozen exact camera contract.
