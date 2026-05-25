# Versioning

The single source of truth for the project version is the most recent
`vX.Y.Z` **git tag**. Both the C++ library and the Python `ag-viz`
package derive their version from it; nothing is hardcoded in source.

This document explains how the system works so that maintainers can cut
releases confidently and contributors understand what they will see
during local development.

## TL;DR

- **Maintainers cutting a release:** push a `vX.Y.Z` tag. That's it.
  Do not edit any version constants by hand.
- **Contributors building locally:** your binary reports something like
  `1.1.3-4-gabc1234-dirty`. That's expected, and it's useful for bug
  triage.
- **Consumers of release tarballs:** the version is baked in. They
  don't need git or any extra tools.

## The release flow

The release process is unchanged from before — you still tag and push,
and [Release Drafter](https://github.com/release-drafter/release-drafter)
still computes the next version from PR labels (`major`, `minor`,
`patch`, etc.; see [CONTRIBUTING.md](../CONTRIBUTING.md#changelog-automation)).
What changed is what happens *after* the tag push.

1. Merge PRs into `main` with appropriate labels.
2. Review the draft release notes that Release Drafter maintains.
3. When ready, tag and push:

   ```bash
   git tag v1.2.3 && git push origin v1.2.3
   ```

4. [`.github/workflows/release.yml`](../.github/workflows/release.yml)
   fires on the tag push. It:
   - Extracts `1.2.3` from the ref.
   - Writes a `VERSION` file at the repo root (a fallback for source
     consumers — see "Source tarballs" below).
   - Configures CMake. `cmake/GetVersion.cmake` runs
     `git describe --tags --match 'v*.*.*' --dirty`, gets `v1.2.3`
     back, strips the `v`, and exposes the result to the build.
   - `configure_file` writes `include/ag/Version.hpp` into the build
     directory with `ag::kVersion = "1.2.3"`.
   - The library and CLI consume that constant. `./ag --version`
     prints `1.2.3`. Saved-model JSON metadata embeds `1.2.3`.
   - The artifact is packaged and the GitHub Release is published.

**You never edit `CMakeLists.txt`, `main.cpp`, or any
`__version__ = "..."` line.** The tag is the only place a version is
declared.

## How the version is resolved

`cmake/GetVersion.cmake` defines `ag_get_version()`, which is called
from the root `CMakeLists.txt` *before* the `project()` declaration.
It tries each source in order:

1. **Git tag.** `git describe --tags --match 'v*.*.*' --dirty` in the
   repo root. On a tagged commit this returns `v1.2.3`; between tags
   it returns `v1.2.3-4-gabc1234` (4 commits past, short sha
   `abc1234`); with uncommitted changes a `-dirty` suffix is appended.
2. **`VERSION` file** at the repo root. Written by the release
   workflow into source tarballs. Used when `.git` is absent.
3. **`0.0.0-unknown`.** Final fallback so configure never fails.

It exposes two variables:

| Variable             | Example                       | Used for                            |
| -------------------- | ----------------------------- | ----------------------------------- |
| `AG_VERSION_FULL`    | `1.2.3-4-gabc1234-dirty`      | `ag::kVersion`, CLI `--version`, JSON metadata |
| `AG_VERSION_NUMERIC` | `1.2.3`                       | `project(VERSION ...)`, library SOVERSION |

CMake's `project(VERSION ...)` is strict: it only accepts `x.y.z`. The
numeric variant strips any suffix so `project()` always succeeds.

The full string is what end users see (`./ag --version`) and what gets
embedded in saved models. Between tags, that string carries enough
information to identify the exact build — invaluable for bug reports.

## How the C++ side consumes it

A small header is generated at configure time:

- Template: [`cmake/Version.hpp.in`](../cmake/Version.hpp.in)
- Generated: `build/<preset>/generated/ag/Version.hpp`
- Exposed via `target_include_directories(arimagarch PUBLIC ...)` in
  [`src/CMakeLists.txt`](../src/CMakeLists.txt) so every consumer of
  the library gets it.

Two constants live in `namespace ag`:

```cpp
namespace ag {
inline constexpr const char* kVersion        = "1.2.3";  // or dev string
inline constexpr const char* kVersionNumeric = "1.2.3";
}
```

Consumers:

- [`src/cli/main.cpp`](../src/cli/main.cpp) wires `ag::kVersion` into
  the CLI `--version` flag.
- [`src/io/Json.cpp`](../src/io/Json.cpp) uses `ag::kVersion` in
  the default `ModelMetadata` constructor, so every saved model
  records the exact build that produced it.

If you add a new place that needs the version, include
`"ag/Version.hpp"` and read `ag::kVersion`. Do not introduce new
hardcoded strings.

## How the Python side consumes it

[`python/pyproject.toml`](../python/pyproject.toml) uses
[setuptools-scm](https://setuptools-scm.readthedocs.io/):

```toml
[build-system]
requires = ["setuptools>=64", "setuptools-scm>=8", "wheel"]

[project]
dynamic = ["version"]   # version comes from setuptools-scm, not from this file

[tool.setuptools_scm]
root = ".."             # pyproject.toml lives in python/; git root is one up
tag_regex = '^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$'
version_file = "ag_viz/_version.py"
fallback_version = "0.0.0+unknown"
```

At install/build time, setuptools-scm reads the git tag and writes
`python/ag_viz/_version.py` (gitignored). At import time,
[`python/ag_viz/__init__.py`](../python/ag_viz/__init__.py) reads
`__version__` from that file, with a fallback to `importlib.metadata`
for installed packages where the source file isn't on disk.

setuptools-scm produces PEP 440 versions:

- At a tag: `1.2.3`
- Between tags with a clean tree: `1.2.4.dev4+gabc1234`
- Between tags with a dirty tree: `1.2.4.dev4+gabc1234.d20260525`

These are normal and expected for development installs.

## What contributors see

After cloning and building locally, the binary's version reflects
whatever your git checkout looks like:

| State                          | `./ag --version` output           |
| ------------------------------ | --------------------------------- |
| Exactly at a tag, clean tree   | `1.1.3`                           |
| 4 commits past, clean tree     | `1.1.3-4-gabc1234`                |
| 4 commits past, dirty tree     | `1.1.3-4-gabc1234-dirty`          |
| No `.git`, no `VERSION` file   | `0.0.0-unknown`                   |

For `ag-viz`, `import ag_viz; ag_viz.__version__` reflects the same
state in PEP 440 form.

This means: **if you file a bug report, the version string in your
output uniquely identifies your build.** Please include it.

## Edge cases and gotchas

### Re-tagging while a build directory exists

CMake doesn't automatically re-run configure when a new git tag is
created. `cmake/GetVersion.cmake` adds `.git/HEAD` and
`.git/packed-refs` to `CMAKE_CONFIGURE_DEPENDS`, which catches most
cases — but lightweight tags created without affecting `packed-refs`
may not trigger a reconfigure. If you create a tag and the version
string doesn't update, run:

```bash
make reconfigure        # via the project Makefile
# or
cmake --preset ninja-release
```

The CI release workflow always configures from a clean checkout, so
this only affects local builds.

### Source tarballs (no `.git`)

GitHub source tarballs (the auto-generated zip/tar.gz on the Releases
page, or anything from `git archive`) lack a `.git` directory. To
support those:

- The release workflow writes a `VERSION` file at the repo root before
  configuring. If you ever produce a source tarball manually, include
  that file.
- `cmake/GetVersion.cmake` reads `VERSION` when git is unavailable.
- For Python, setuptools-scm bakes `_version.py` into any built sdist
  or wheel, so installed consumers don't need git.

### `workflow_dispatch` runs

The release workflow can be triggered manually
(`workflow_dispatch`). In that case there is no tag, and the workflow
substitutes `0.1.0` as a placeholder version for the build. The
resulting artifact is not published as a release. This path exists
only for verifying that the workflow itself works; do not rely on its
version output.

### PEP 440 local segments in the Python version

Between tags, setuptools-scm produces versions like
`1.2.4.dev4+gabc1234`. The `+gabc1234` is a PEP 440 "local version
segment." Most package indexes (including PyPI) reject local
segments. This is fine for local development installs but would need
attention if `ag-viz` is ever published to PyPI — at that point, the
release workflow should build the sdist from a tagged commit (so the
version is clean) rather than from arbitrary HEAD.

## Files involved

| File                                                                 | Role                                                       |
| -------------------------------------------------------------------- | ---------------------------------------------------------- |
| [`cmake/GetVersion.cmake`](../cmake/GetVersion.cmake)                | Resolves version from git/VERSION file/fallback            |
| [`cmake/Version.hpp.in`](../cmake/Version.hpp.in)                    | Template for the generated `ag/Version.hpp`                |
| [`CMakeLists.txt`](../CMakeLists.txt)                                | Calls `ag_get_version()` and `configure_file()`            |
| [`src/CMakeLists.txt`](../src/CMakeLists.txt)                        | Exposes the generated include dir; installs `Version.hpp`  |
| [`src/cli/main.cpp`](../src/cli/main.cpp)                            | Reads `ag::kVersion` for the `--version` flag              |
| [`src/io/Json.cpp`](../src/io/Json.cpp)                              | Reads `ag::kVersion` for saved-model metadata              |
| [`python/pyproject.toml`](../python/pyproject.toml)                  | Configures setuptools-scm                                  |
| [`python/ag_viz/__init__.py`](../python/ag_viz/__init__.py)          | Reads `__version__` from generated `_version.py`           |
| [`.github/workflows/release.yml`](../.github/workflows/release.yml)  | Writes `VERSION` file before configure                     |

## If something looks wrong

- `./ag --version` shows `0.0.0-unknown`: the build is missing both
  `.git` and a `VERSION` file. If this is a CI build, check the
  "Write VERSION file" step. If local, you probably built from an
  exported tarball — drop a `VERSION` file at the repo root.
- Version doesn't update after a new tag locally: run `make reconfigure`.
- `ag_viz.__version__` is `0.0.0+unknown`: setuptools-scm couldn't
  find a tag *and* `_version.py` wasn't generated. Re-install with
  `make py-install` (or `pip install -e python/`) from inside a git
  checkout.
