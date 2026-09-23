# Agent SDF — vision

Build a lightweight pseudo-CAD workspace where an agent writes a scene in our GLSL-like Python DSL inspects chosen cameras, and edits the geometry again.

Our DSL lowers to GLSL describing a set of SDF objects: Scene -> list[Objects]

Ordered boolean logic is implemented using..
`min(..)` `max()`
And everything is assumed to be vectorized


## Modeling vocabulary

`Shape` records primitives, transforms, paths, and CSG as an expression tree. `Model` records ordered, named CSG steps and their dependencies. `Field` and `custom_sdf` let authors express geometry that the vocabulary cannot yet describe.

Observe agent-authored scenes and add stable operations to `Shape` as they become useful. Lofts and variable-thickness extrusions in `jet_fighter.py` are candidates.
