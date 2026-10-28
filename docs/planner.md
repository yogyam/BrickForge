# Assembly planner (M2)

Rules (R1): one piece per step, upright insertion straight down from above, horizontal work surface, start
from a ground piece. A piece can be added when it does not collide, attaches to the assembled component
(ground contact or studs of a placed piece below it), its vertical insertion path is clear of placed bodies,
and it does not seal an unplaced piece beneath it.

The last rule is an exact dead-end test: once a piece is covered from above it can never be inserted, so
placing the cover first is always wrong. With that test, greedy selection is complete. If any full order
exists, the first unplaced piece of that order is always attachable and safe, so the planner never gets
stuck and backtracking is unnecessary. A node and time budget still exist so the `planning_unknown` outcome
is honoured, but with at most 100 pieces the budget is never reached in practice.

Tie-break between candidates: lowest `z`, then most placed supporters, then id. For upright bricks of one
height this yields a layer-by-layer order.

Outcomes: `planned` (steps stored on the build and replayed by the validator, which also checks the
insertion path), `planning_failed` (no order under these rules, remaining pieces listed; this is a limit of
the insertion model, not a proof of physical unbuildability), `planning_unknown` (budget exhausted).

Consequence worth recording for the writeup: for the R1 catalog (same-height upright bricks, every
non-ground piece supported from below), every structurally valid build has a planned order, so planning
can only fail once plates, varying heights or sideways parts are introduced. A build with no checked
steps is still inspectable and exports without `STEP` markers, labelled as such.

Not modelled: hand clearance, insertion force, stud clutch, gravity. See `docs/stability.md` once M2.5 lands.
