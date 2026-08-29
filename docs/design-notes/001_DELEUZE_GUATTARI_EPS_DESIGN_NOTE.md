# Design Note 001 — Percept, Affect, Concept: Philosophical Provocations for Ecological Predictive States

```text
Status: CANONICAL NON-AUTHORITATIVE DESIGN EXPLORATION
Authority: NONE
Scientific review: SCIENTIFIC_PASS at exact implementation head 4e37f9cc25a33927135077a76cf72eae9d191392
Scientific result: NONE
Benchmark effect: NONE
Gate effect: NONE
```

The exact-head scientific pass and owner-approved closeout make this note canonical only as a
non-authoritative design exploration; they do not scientifically support or authorise its
hypotheses.

## 1. Purpose

This note records possible architectural and research implications of Gilles Deleuze and Félix
Guattari's chapter **“Percept, Affect, and Concept”** from *What Is Philosophy?* for the wider
Ecological Predictive States (EPS) and Unfrozen Schemas programmes.

It is deliberately non-authoritative.

It does not modify or reinterpret:

- `docs/RESEARCH_CHARTER.md`;
- `docs/EPS_BENCH_V0.md`;
- any milestone or gate specification;
- any benchmark label, schema, identity, threshold, seed, or split;
- any reviewed scientific claim;
- any implementation or review disposition.

The chapter is not an empirical theory of perception or a robotics specification. Its concepts
should therefore be used to generate **candidate architectural hypotheses**, which must later be
translated into explicit, measurable, falsifiable scientific or engineering contracts before they
can influence implementation.

## 2. Source distinction: Deleuze and Guattari are not Gibson

Deleuze and Guattari's **percept** is an aesthetic being of sensation: it is no longer a perception
belonging to a perceiving subject (Deleuze and Guattari 1994, 164–165). Their **affect** is a
nonhuman becoming and a zone of indetermination or indiscernibility, not a personal feeling or
lived-state report (172–173). The work of art creates a compound of percepts and affects capable of
“standing up” on its own (164–165). The chapter also develops landscape and nonhuman becoming
through a becoming with the world (168–169).

Gibson's ecological theory of perception is an empirical account of information available to an
organism in an environment (Gibson 1979). Gibsonian invariants, ambient optic structure,
affordances, and perception-action coupling belong to a different intellectual and evidential
project.

The useful relationship is therefore neither identity nor derivation:

```text
Deleuze and Guattari:
    philosophical provocation concerning sensation, composition,
    material expression, becoming, and heterogeneous forms of thought

Gibson:
    empirical ecological psychology of organism-environment perception

EPS:
    controlled scientific apparatus for testing ecological perceptual structure

Unfrozen Schemas:
    typed coordination and provenance across heterogeneous agents and claims
```

Any operational analogy introduced below must preserve this distinction.

## 3. Terminology safeguards

| Source term | Source role in the chapter | Possible operational inspiration | Equation that must not be made |
| --- | --- | --- | --- |
| Percept | A self-sustaining aesthetic being of sensation, no longer a subject's perception (Deleuze and Guattari 1994, 164–165) | A typed ecological relational record that is not reducible to a private picture | `percept = EPS label` |
| Affect | A nonhuman becoming and zone of indetermination or indiscernibility, not a personal feeling or lived-state report (Deleuze and Guattari 1994, 172–173) | A future `CAPACITY_TRANSITION` record of changing agent-environment capacities | `affect ≠ CAPACITY_TRANSITION`, `affect ≠ affordance`, `affect ≠ reward`, and `affect ≠ emotion score` |
| Plane of composition | The compositional field on which sensations and their compounds stand and open onto further forces (Deleuze and Guattari 1994, 190–192) | A protocol for composing heterogeneous cognitive records | `plane of composition = latent space` |
| Frame or section | An interface or face of a compound, not merely a coordinate (Deleuze and Guattari 1994, 178–181, 186–187) | Surface, opening, boundary, attachment, interface, or transition structure | `frame = bounding box` |
| Counterpoint | A relational composition in which heterogeneous beings or trajectories answer one another (Deleuze and Guattari 1994, 184–186) | Body-relative affordance and morphology-environment coupling | `counterpoint = similarity metric` |
| Heterogenesis | Distinct forms of thought call for further heterogeneous creations without becoming one synthesis (Deleuze and Guattari 1994, 196–199) | Typed neural-colony architecture preserving distinct epistemic products | `heterogenesis = ensemble voting` |
| Rhizome / assemblage | Broader concepts invoked from *A Thousand Plateaus* rather than chapter 7 of *What Is Philosophy?* (Deleuze and Guattari 1987) | Distributed systems as one limited analogy | `cloud computing ≠ rhizome` |
| Deterritorialization / reterritorialization | Concepts used across the broader Deleuze–Guattari corpus and within “Percept, Affect, and Concept,” including its movement from house or territory toward universe and plane of composition (Deleuze and Guattari 1994, 181, 185, 197–198) | A limited question about how systems open beyond or reconcentrate local authority | `network distribution ≠ deterritorialization` |

## 4. From an internal picture to an ecology of relations

One of the chapter's strongest provocations for EPS is its description of the **house** through
differently oriented sections and joined planes (Deleuze and Guattari 1994, 178–181):

```text
foreground and background
horizontal and vertical
left and right
straight and oblique
walls, floors, doors, windows, mirrors, and openings
```

The house is a finite junction of planes that can open toward a larger field of forces (178–181).
Later, the chapter says that frames or sections are not coordinates: they are faces and interfaces
of a compound (186–187).

This does not scientifically establish an ecological robotics ontology. It nevertheless sharpens a
design alternative to the idea that a robot must first reconstruct one exhaustive Cartesian scene
before it can perceive anything meaningful.

A possible EPS-oriented primary description is:

```text
surfaces
orientations
boundaries
local attachments
occluding ownership
openings
supports
disclosures and concealments
transport under action
agent-relative trajectories
```

Metric coordinates remain indispensable instruments. They may be used for apparatus generation,
validation, navigation, manipulation, mapping, and communication. The design claim is narrower:

> Cartesian coordinates need not be the universal representational substance into which every
> perceptual relation is first translated.

The current EPS apparatus documented in the [Slice 3 analytic-flow
contract](../GATE_0B_SLICE_3_ANALYTIC_FLOW.md), [Slice 4 boundary/event
contract](../GATE_0B_SLICE_4_BOUNDARY_EVENTS.md), and [Slice 5 appearance-candidate
contract](../GATE_0B_SLICE_5_APPEARANCE_CANDIDATES.md) already points in this direction through:

```text
opaque surfaces
analytic optical transport
oriented boundary ownership
local projected attachment loci
accretion and deletion
frame entry and exit
occlusion relations
appearance-candidate audit evidence with no admitted profile
```

These are not Deleuzian percepts. They are scientifically controlled relational constructs that
happen to make the philosophical alternative easier to articulate.

## 5. Composition rather than universal representation

The chapter repeatedly treats **composition** as more basic than representation (Deleuze and
Guattari 1994, 190–192). Its artistic claim should not be copied into science, but it suggests a
powerful engineering question:

> Should embodied intelligence be organized around one universal world representation, or around
> the composition of several distinct but interoperable epistemic products?

EPS and Unfrozen Schemas increasingly favour the second option.

A possible architecture would preserve at least five registers:

```text
1. Sensory records
   RGB, depth where permitted, tactile, audio, proprioception

2. Ecological percept records
   surfaces, transport, boundaries, visibility, occlusion, openings

3. Scientific-function records
   metric estimates, dynamics, uncertainty, simulation, calibrated prediction

4. Capacity-transition records
   changes in reachability, traversability, inspectability, support, risk, or disclosure

5. Conceptual records
   language-level hypotheses, explanations, plans, counterfactual questions
```

None should silently overwrite the others.

A metric estimator may state:

```text
estimated opening width: 0.82 m
uncertainty interval: ...
```

while an ecological record states:

```text
opening persists under forward motion
left and right surfaces bound a traversable candidate interval
```

and an LLM states:

```text
hypothesis: this may be a doorway
evidence references: ...
status: unverified
```

The conceptual hypothesis must not replace the ecological or metric evidence from which it was
derived.

## 6. Selective invariance, not material blindness

The chapter treats material as passing into expressive sensation rather than as a neutral carrier
that can always be discarded (Deleuze and Guattari 1994, 166–167). Colour, line, light, texture,
sound, and material therefore cannot be cleanly separated from expression. The percept can make
otherwise imperceptible forces perceptible: weight, rotation, pressure, expansion, time, growth,
and mechanical movement (182–183).

This offers an important correction to a crude interpretation of ecological abstraction.

The objective should not be:

```text
remove appearance until only abstract geometry remains
```

Instead, future EPS work should distinguish:

### 6.1 Nuisance appearance changes

These alter sensory presentation without changing the ecological situation under study:

- arbitrary palette changes;
- controlled texture-family and frequency changes;
- bounded illumination changes;
- renderer-specific raster differences;
- style-slot permutation independent of surface role.

[Slice 5](../GATE_0B_SLICE_5_APPEARANCE_CANDIDATES.md) audits candidate controls in this class; it
does not freeze an appearance profile.

### 6.2 Ecologically expressive material changes

These alter or reveal action-relevant properties:

- wet versus dry support;
- transparent versus opaque barrier;
- flexible versus rigid surface;
- hot versus cool object;
- corroded versus sound structure;
- loaded deformation;
- surface flow revealing an opening;
- social or warning colour;
- biological ripeness or decay.

A good ecological model should **not** remain invariant to all of these changes.

A future research programme should therefore test both:

```text
Nuisance invariance:
    the ecological state remains stable when the action-relevant situation is unchanged

Expressive-material sensitivity:
    the ecological state or its uncertainty changes when material appearance reveals a new
    force, capacity, risk, or affordance
```

A concise design rule is:

> The goal is not to escape pixels, but to stop treating pixels merely as pixels. They can be
> material carriers of surfaces, forces, events, and possibilities for action.

## 7. From visibility events to capacity transitions

Deleuze and Guattari's affect should not be implemented as a robot's emotion, reward value,
affordance probability, or capacity-transition record (Deleuze and Guattari 1994, 172–173).

The project's candidate engineering translation—not the philosophical definition—is a
transformation of an **agent-environment capacity**:

```text
an opening becomes traversable
a surface becomes reachable
a grasp becomes available
a support becomes unstable
a route becomes blocked
a concealed region becomes inspectable
a collision becomes imminent
a previously available action becomes impossible
```

This suggests a future layer above the present EPS apparatus:

```text
Ecological percept:
    what relational structure is present?

Capacity transition:
    what possibilities for this embodied agent were gained, lost, strengthened,
    weakened, disclosed, or concealed?
```

For example:

```text
EPS visibility event:
    region C becomes optically disclosed after action A

Future capacity consequence:
    a passage hypothesis can now be tested
```

The first is a present perceptual construct. The second would require a later affordance/action
contract and must not be smuggled into current visibility-event labels.

A neutral technical name such as `CAPACITY_TRANSITION` is preferable to naming a schema
`AFFECT`, because the philosophical and engineering concepts are not identical.

## 8. Ambiguity as a first-class epistemic state

The chapter treats zones of indetermination and indiscernibility as productive rather than merely
defective (Deleuze and Guattari 1994, 172–173). This resonates with an existing EPS discipline: do
not convert every unresolved state into a fabricated positive or zero-valued conclusion.

An embodied system should distinguish:

```text
known empty
known present
currently unobservable
not derivable under this apparatus
locally ambiguous
contradictory evidence
unsupported capability
```

Existing EPS examples include:

```text
MULTI_SURFACE_JUNCTION_AMBIGUOUS
ANALYTIC_BOUNDARY_AMBIGUOUS
UNRESOLVED_OCCLUSION
known-empty versus unavailable occlusion
component topology unavailable
```

This is not an argument for romanticizing errors. Some ambiguity is caused by faulty apparatus,
insufficient resolution, or implementation defects and should be corrected. The design principle is
that epistemic status must remain typed and traceable.

An attached LLM must preserve that status. It must not turn:

```text
boundary ambiguous
```

into:

```text
this is probably a door
```

and then allow the latter to re-enter the perceptual state as fact.

## 9. Counterpoint and body-relative affordance

The chapter's discussion of Uexküll describes nature contrapuntally: web and fly, tick and mammal,
shell and hermit crab, organism and environmental trajectory (Deleuze and Guattari 1994, 184–186).
The relationship is neither a property of one isolated object nor a teleological master plan.

A robotics analogue is:

> An affordance model without a body and action model is incomplete.

The same physical opening may be:

```text
traversable for a narrow wheeled robot
non-traversable for a wider robot
reachable for a long manipulator
unreachable for a short manipulator
safe for a slow agent
unsafe for a fast agent with a long stopping distance
```

The environmental percept need not be rebuilt from nothing for every body. Instead, one ecological
description may compose with different morphology and skill records to produce different capacity
structures.

A useful experiment would hold the visual environment constant while changing:

```text
body width
turning radius
camera height
manipulator reach
available actions
current payload
```

The research question would be:

> Can a stable environmental relational representation support different agent-relative capacity
> structures without collapsing environment and body into one opaque latent state?

## 10. The Gibsonian perceiver and the LLM reasoner

The chapter's closing distinction among art, science, and philosophy is especially useful for the
Gibsonian-head/LLM architecture. The three forms of thought intersect and call for one another,
without becoming one synthesis; their intersection instead calls for further heterogeneous
creation (Deleuze and Guattari 1994, 196–199).

The engineering analogue is not a mapping of philosophy directly onto modules. It is a rule of
epistemic separation:

```text
Ecological perceiver:
    emits relational percept records

Scientific estimator:
    emits functions, measurements, uncertainties, and calibrated predictions

Capacity reasoner:
    emits agent-relative possibility and transition records

LLM conceptual reasoner:
    emits hypotheses, abstractions, explanations, plans, and questions

Composition protocol:
    preserves type, provenance, authority, contradiction, and supersession
```

The LLM should not be the sovereign location in which every other representation is dissolved into
language.

It may:

- formulate hypotheses;
- ask for missing evidence;
- connect observations across time;
- propose counterfactual actions;
- explain typed records to humans;
- coordinate specialist calls;
- reason over longer horizons.

It may not silently:

- rewrite a perceptual record;
- convert uncertainty into fact;
- infer metric truth from prose;
- treat a plan as an observation;
- treat model consensus as scientific validation;
- override local safety or immediate ecological evidence.

## 11. A typed epistemic bus

A neural colony inspired by this note should communicate through typed records, not unrestricted
natural-language messages.

A possible record family is:

```text
SENSORY_OBSERVATION
ECOLOGICAL_PERCEPT
SCIENTIFIC_ESTIMATE
CAPACITY_STATE
CAPACITY_TRANSITION
CONCEPT_HYPOTHESIS
COUNTERFACTUAL_QUERY
PLAN_PROPOSAL
CONTRADICTION
VERIFICATION_RESULT
UNAVAILABLE_CAPABILITY
```

Each record should carry, where applicable:

```text
source agent or apparatus
input record references
exact model/code/schema version
identity or content hash
permission class
epistemic status
uncertainty
authority class
validity interval
supersession links
```

The colony should permit disagreement. A conceptual record may conflict with an ecological record,
but it may not overwrite it. Resolution requires an explicit verification or adjudication process.

This is more than an ensemble of models voting over one answer. It is the composition of
heterogeneous products whose differences remain operationally meaningful.

## 12. Unfrozen Schemas and heterogeneous composition

Unfrozen Schemas provides a natural substrate for this architecture because it treats typed
interfaces, provenance, role separation, and bounded authority as central rather than incidental.

The [EPS review workflow](../review-protocol.md) is itself a useful organizational analogy:

```text
implementation
engineering review
scientific review
owner judgement
closeout
empirical gate evaluation
```

The workflow's explicit separation of these roles illustrates the architectural analogy.
Engineering review addresses defects that scientific review may not. Scientific review addresses
construct errors that passing tests may not. Owner approval does not become empirical evidence.
Closeout does not repair implementation.

This organizational analogy should not be confused with a scientific result about cognition. It
does, however, motivate one architectural intuition:

> Composition of heterogeneous authorities can be more reliable and auditable than premature
> unification.

## 13. Opinion and chaos as engineering failure modes

The chapter's final warning identifies two dangers when heterogeneous forms of thought intersect:
return to opinion and collapse into chaos (Deleuze and Guattari 1994, 196–199).

These have useful AI analogues.

### 13.1 Return to opinion

The conceptual reasoner turns evidence into a familiar story and then forgets the distinction:

```text
ambiguous opening
→ “probably a doorway”
→ doorway treated as observed fact
```

Controls include:

- typed epistemic status;
- evidence references;
- no silent percept mutation;
- contradiction records;
- independent verification;
- bounded authority.

### 13.2 Collapse into chaos

A distributed multi-agent system generates incompatible messages, representations, plans, and
partial truths with no stable composition protocol.

Controls include:

- strict schemas;
- versioned identities;
- routing and authority rules;
- capability declarations;
- provenance;
- explicit supersession;
- bounded coordinator authority.

This gives a practical role to Unfrozen Schemas: prevent a neural colony from becoming either one
LLM's opinion or an uncoordinated swarm.

## 14. Cloud and distributed systems: analogy and limit

As a limited design analogy, rhizome and assemblage from *A Thousand Plateaus* (Deleuze and Guattari
1987), together with the wider vocabulary of deterritorialization and reterritorialization—which
also appears in “Percept, Affect, and Concept” in passages connecting house or territory, universe,
and plane of composition (Deleuze and Guattari 1994, 181, 185, 197–198)—can be brought into
conversation with later distributed systems:

```text
distributed identity
nonlinear communication
modular services
temporary assemblages
dynamic recombination
multiple overlapping communities
flows detached from one physical location
```

This does not mean Deleuze and Guattari technically predicted cloud computing.

Cloud systems also create centralisation and reterritorialisation:

```text
hyperscale providers
identity systems
billing boundaries
API control
proprietary model endpoints
central observability
jurisdiction
data ownership
```

A system may look distributed at the application level while remaining highly concentrated
institutionally and physically.

For cloud robotics, the design implication is:

```text
On-device ecological authority:
    immediate perception-action coupling
    local safety
    local boundary, visibility, and capacity records

Remote or cloud contribution:
    conceptual reasoning
    large memory
    cross-robot knowledge
    long-horizon planning
    language interaction
```

Disconnection should not remove the robot's basic ability to perceive and act safely. The cloud may
extend the local territory toward a larger conceptual field, but it should not be the sole source of
the robot's reality.

## 15. Candidate research hypotheses

The ideas above should enter science only through explicit future work packages.

### 15.1 Selective appearance invariance

Compare performance under:

```text
nuisance appearance transformations
ecologically expressive material transformations
ambiguous transformations
```

Hypothesis:

> A relational ecological model can remain invariant to nuisance appearance while changing state
> appropriately when material appearance reveals action-relevant forces or capacities.

### 15.2 Percept-to-capacity transitions

Add a bounded action and body contract above the present EPS apparatus.

Hypothesis:

> Explicit ecological percepts plus typed capacity transitions generalize more robustly than
> pixels or object labels alone under appearance and viewpoint shifts.

### 15.3 Multi-body counterpoint benchmark

Use one environment with multiple robot morphologies.

Hypothesis:

> A shared environmental relational representation can compose with different body/action models
> to produce different affordance structures without reconstructing a separate complete world for
> each agent.

### 15.4 Typed neural-colony benchmark

Compare:

```text
monolithic reasoner
untyped multi-agent messages
typed heterogeneous colony
```

on:

- contradiction containment;
- appearance OOD;
- missing-module robustness;
- network disconnection;
- explanation fidelity;
- hallucination containment;
- provenance preservation.

Hypothesis:

> Preserving heterogeneous epistemic products through typed composition improves robustness and
> auditability relative to forcing all products into one latent or linguistic representation.

### 15.5 Local ecological authority under cloud interruption

Test a robot with and without remote LLM/cloud connectivity.

Hypothesis:

> Immediate ecological perception and safety remain functional under network loss, while the cloud
> improves conceptual interpretation and long-horizon coordination without becoming perceptually
> authoritative.

## 16. Falsification and disconfirmation

This note should not become an interpretive shield that explains every result.

The programme should be willing to find that:

- Cartesian or object-centric representations outperform ecological representations;
- one integrated model outperforms heterogeneous composition;
- capacity-transition records add no measurable value;
- typed colonies create overhead without robustness benefit;
- appearance distinctions proposed here are not learnable or operationally stable;
- cloud/local separation harms performance;
- the philosophical analogies fail to generate useful experiments.

Such outcomes would challenge the proposed engineering translations, not prove or disprove
Deleuze and Guattari's philosophy.

## 17. Non-goals

This note does not authorize:

- renaming EPS records as percepts or affects;
- adding philosophical terminology to public benchmark schemas;
- treating aesthetic theory as empirical evidence;
- changing Slice 1–5 contracts;
- changing benchmark profiles, seeds, thresholds, or identities;
- benchmark freezing;
- Gate 0C or Gate 0D;
- model implementation;
- LLM integration;
- cloud deployment;
- robot hardware;
- a scientific-result claim.

## 18. Suggested sequencing

A disciplined sequence is:

```text
1. Keep this note non-authoritative.
2. Extract one candidate hypothesis at a time.
3. Draft a separate scientific and engineering specification.
4. Review construct validity before implementation.
5. Implement under the ordinary exact-head workflow.
6. Preserve negative outcomes.
7. Advance authority only through explicit owner and gate decisions.
```

The first likely future bridge from the current EPS programme is not an LLM. It is a bounded
`CAPACITY_TRANSITION` experiment built on the already canonical perceptual apparatus.

## 19. References

Inline citations use author and year. Printed page numbers are supplied for page-specific claims
from the consulted chapter; broader work-level references may cite author and year only. The
broader philosophical source remains separate from chapter 7, and the distributed-systems analogy
is not a claim of either source.

### Primary chapter source

- Deleuze, Gilles, and Félix Guattari. *What Is Philosophy?* Translated by Hugh Tomlinson
  and Graham Burchell. New York: Columbia University Press, 1994. Chapter 7, “Percept, Affect, and
  Concept,” pp. 163–199.

### Broader philosophical source

- Deleuze, Gilles, and Félix Guattari. *A Thousand Plateaus: Capitalism and Schizophrenia*.
  Translation and foreword by Brian Massumi. Minneapolis: University of Minnesota Press, 1987.

### Related scientific source

- Gibson, James J. *The Ecological Approach to Visual Perception*. Boston: Houghton Mifflin, 1979.

### Internal project documents

- [Gate 0B Slice 3 — Analytic Optical Transport](../GATE_0B_SLICE_3_ANALYTIC_FLOW.md)
- [Gate 0B Slice 4 — Oriented Boundaries and Visibility Events](../GATE_0B_SLICE_4_BOUNDARY_EVENTS.md)
- [Gate 0B Slice 5 — Procedural Appearance Candidates](../GATE_0B_SLICE_5_APPEARANCE_CANDIDATES.md)
- [Research-software review protocol](../review-protocol.md)

The relationship between these sources is interpretive and exploratory. No direct historical or
scientific derivation is asserted.
