# Prospective EPS capture diagnostic

Review profile: `DUAL_REVIEW`  
Evidence class: `PUBLIC_REPOSITORY_ONLY`

This helper implements only the bounded ordinary-development diagnostic advised
after the retained original corridor audit. It does not change a frozen capture
contract, labels, scene topology, accepted configuration, or production
renderer.

## Fixed matrix and stop conditions

The only cells are WGL and OSMesa at requested `model.vis.quality.offsamples`
4, followed by WGL and OSMesa at 0. There is one attempt per cell, no retry,
and no expansion. Before a callback can create a renderer context it must
verify the retained receipt/manifest bindings, needed files, origin metadata,
source and seeds, resolved configuration, compiled geometry, cameras, and
appearance against the original episode-0 evidence.

The callback must compile the existing reviewed corridor with existing builders
and appearance resolution, set only `model.vis.quality.offsamples` before
context creation, and capture RGB, depth, and segmentation in the historical
call order. It must retain the segmentation RGB passed by `Renderer.render(out=uint8[H,W,3])`
before decode, the returned decoded segmentation, and the actual scene
`segid -> (objid, objtype)` mapping. The helper accepts a raw geom-ID map
derived from that recorded scene map; it never assumes raw IDs equal segids.

Both requested-4 baseline cells must exactly reproduce the retained original
decoded arrays before either requested-0 callback begins. An input mismatch,
capture error, unavailable material provenance, decode mismatch, or baseline
failure stops the matrix. Partial outputs and exception receipts are retained;
existing ledger paths are never overwritten.

Every successful cell records requested and actual samples, attachment facts
when supported, GL vendor/renderer/version, model/XML/package/binary hashes,
the full map, encoded RGB, decoded output, and control captures. Unsupported
material introspection is recorded as explicit unknown or causes failure; it
must not silently fall back. Any GL-binding inspection must restore bindings in
a `finally` block.

No function in this helper imports MuJoCo or creates a graphics context. The
future capture callback is deliberately supplied by a separately reviewed,
owner-directed execution environment. This implementation PR is not authority
to run the diagnostic.
