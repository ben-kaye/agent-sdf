We are creating an expressive Python geometry API for building, combining, and inspecting 2D and 3D shapes, with CAD-like spatial operations and procedural geometry as first-class features.

Geometry at its lowest level is defined with fields.

# Working rules

- Keep documentation extremely concise, well laid out, and focused on a small purpose.
- When implementing code, do not modify documentation.
- Use subagents when appropriate, and select a Sol-class model for them.
- Challenge decisions that appear unsound. Suggest foundational changes when warranted; everything is subject to change
- Public APIs must be discussed. They should be convenient, explanatory and concise. Ideally, the Schelling point of intent.


## Conventions

- Use right-handed, Z-up world axes.
- The camera looks along local -Z, with local +Y up and right-handed local axes.
- Write quaternions as `(w, x, y, z)`.

## Implementation

- Do not add persistent tests, inspect `tests/`, or run `tests/`.
- Do not add compatibility support. Migrate examples to the latest API when needed.
- Do not reference documentation in code.
- On completion, run `ruff format .` and address all findings from `ruff check .`.


## Creation / generation

When a user asks you to create geometry, follow these rules:

- Do not inspect `src/`.
- Read `README.md` and use the public API. Compose its operations into high-level procedures as needed.
- Use `custom_sdf` when the public operations cannot express the shape clearly.
- Render often to inspect your geometry. 
- If the documented API is unclear or insufficient, report the specific gap.
- Avoid encoding text in the geometry.