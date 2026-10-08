# A2 reappearance diagnostic v1

This version defines a deterministic source-only candidate and its fixed controls over
`BoundedPrefixMemory`. It consumes three complete, consecutive Boolean mask frames,
two executed commands, and one announced command. It supports pure lateral commands
with exact rational values. Typed access is checked before feature or binding reads.

For each aligned row, the candidate uses the last two visible masks. It estimates
horizontal image velocity as the change in column centroid divided by the summed
lateral commands between those observations. It translates the last mask to the
current and announced command locations using exact rational arithmetic, rounds
half-column ties away from zero, shifts only columns, and crops at image bounds.
Potential blockers are selected before any future: they are the other currently
visible rows whose masks intersect the query mask transported to the current command
location. Each selected blocker is transported from its own original last visible
mask using its own two-sighting velocity. The candidate returns `q=1` if any support
remains after removing the union of transported blocker masks, and `q=0` otherwise.

If a query or selected blocker lacks two sightings, or its two sightings have a zero
summed-command denominator, the candidate returns `q=1/2`. A missing sighting has no
mask and never becomes a query filter. No floor, foreground, role, raw identifier,
camera pose, depth, future mask, or target label is consulted.

The output contains one row for every aligned memory row, including remembered rows
and rows with insufficient history. Each method returns a canonical rational `q`
string and a diagnostic fallback flag and reason. The fixed methods are:

- `motion_overlap`: candidate transport with preselected blocker subtraction.
- `motion_only`: the same query transport without blocker subtraction.
- `zero`, `one`, and `half`: constant prediction controls.
- `current_presence`: whether the row has a nonempty current mask.
- `toward_last_view`: `1` when the announced command location is strictly closer
  than the current location to the last observed command location, `0` when farther,
  and `1/2` at equal distance or with no sighting.
- `exact_retrace`: the observed presence bit for an exact match to one of the three
  admitted command locations, otherwise `1/2`. If a command location occurs more
  than once, the first prefix frame supplies the bit; the static command contract
  requires duplicate locations to agree.

The canonical output stores binding and feature hashes outside the `q` values. Rows
and blocker diagnostics use numeric positions in the memory alignment and omit opaque
token strings. `selected` lists the row indices selected by the current-mask proxy.
`erased_support` lists each selected blocker’s intersection count with the original
query future mask, independently; counts may overlap. When blocker processing falls
back, counts for processed blockers are present and unprocessed entries are `null`.
The candidate itself subtracts the union, so overlapping counts do not change its
prediction.

This is a fixed untrained diagnostic rule. It is not a learned model, evaluator,
generator, or evidence of predictive advantage. Handwritten mask checks do not
qualify geometry, physical surfaces, or any empirical gate.
