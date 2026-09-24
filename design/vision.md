# Agent SDF — vision

Create an expressive Python geometry API that lets an agent build, combine, and inspect detailed 2D and 3D shapes rapidly.

Geometry is defined by composable fields. Python functions, loops, parameters, and custom fields let the agent describe forms beyond a fixed set of primitives. CAD-like spatial operations relate sketches in 3D planes to solids through placement, extrusion, lofting, and sweeping.

Rendering works from the fields directly, so triangulation and meshing are not required during authoring.

## Fast feedback closes the loop

The agent can run a Python scene, preview chosen camera views, inspect the images and geometry, and revise the source. This feedback loop should be fast and available through a simple command so the agent can iterate on detailed forms autonomously.
