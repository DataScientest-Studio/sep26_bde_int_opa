# scratch/ — personal, never committed

Anything you try out for yourself goes here: test scripts, notebooks, SQL
experiments, data dumps. This whole folder is ignored by git (see `.gitignore`),
so nothing in it will ever show up in a commit or pull request.

Rules
- Shared code goes to `src/`, `sql/`, `tests/`, `docs/`. `scratch/` is only for yourself.
- Never edit someone else's file just to test it: copy it into `scratch/` and test there.
- Scripts here can import the shared code, for example:

    import sys; sys.path.insert(0, "src/data")   # run from the repo root
    from db import get_pg_connection

Extra personal ignores (beyond `scratch/`) go to `.git/info/exclude`, which is
local to your clone and is never pushed.
