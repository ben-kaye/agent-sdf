
# rules
DOCS MUST BE EXTREMELY CONCISE. WELL LAYED OUT. SMALL PURPOSE
if implementing do not modify docs
use subagents if appropriate (class == sol)
question bad decisions
everything is subject to change, do not be afraid to suggest a foundational change

## conventions
z-up, right handed axes
camera follows a -Z ray axis, +Y up right handed local axes
quaternions (w,x,y,z)


## implementation
no persistent tests. do not look at tests/ do not run tests/
no compatibility engineering, migrate examples to latest api if needed
address `ruff check .` on completion of changes and run `ruff format .`