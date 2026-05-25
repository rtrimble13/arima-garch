# GetVersion.cmake
#
# Single source of truth for the arima-garch project version.
#
# The authoritative version is the most recent `vX.Y.Z` git tag.
# This module resolves it at configure time and exposes:
#
#   AG_VERSION_FULL    - Descriptive string, e.g. "1.1.3"
#                        or "1.1.3-4-gabc1234-dirty" between tags
#                        or "0.0.0-unknown" if the version cannot be
#                        determined. Use this for human-facing output
#                        (CLI --version, embedded build metadata).
#   AG_VERSION_NUMERIC - Pure x.y.z prefix, suitable for
#                        project(... VERSION ...) and library SOVERSION.
#
# Resolution order:
#   1. `git describe --tags --match v*.*.* --dirty` in the repo root.
#   2. A `VERSION` file at the repo root (written by the release
#      workflow into source tarballs that have no .git).
#   3. The literal "0.0.0-unknown" fallback.
#
# See docs/versioning.md for the full release workflow.

function(ag_get_version)
  set(_version "")

  find_package(Git QUIET)
  if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --tags --match "v*.*.*" --dirty
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE _raw
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _rc)
    if(_rc EQUAL 0 AND _raw)
      string(REGEX REPLACE "^v" "" _version "${_raw}")
    endif()
  endif()

  # Fallback: VERSION file shipped in source tarballs.
  if(NOT _version AND EXISTS "${CMAKE_SOURCE_DIR}/VERSION")
    file(READ "${CMAKE_SOURCE_DIR}/VERSION" _version)
    string(STRIP "${_version}" _version)
  endif()

  if(NOT _version)
    set(_version "0.0.0-unknown")
  endif()

  # project(VERSION ...) only accepts pure x.y.z. Strip any suffix.
  string(REGEX MATCH "^([0-9]+\\.[0-9]+\\.[0-9]+)" _numeric "${_version}")
  if(NOT _numeric)
    set(_numeric "0.0.0")
  endif()

  set(AG_VERSION_FULL    "${_version}"      PARENT_SCOPE)
  set(AG_VERSION_NUMERIC "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

# Re-run configure when refs change so new tags are picked up.
# Imperfect: lightweight tags created without touching packed-refs
# may require an explicit `make reconfigure`.
if(EXISTS "${CMAKE_SOURCE_DIR}/.git/HEAD")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/.git/HEAD")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/.git/packed-refs")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/.git/packed-refs")
endif()
