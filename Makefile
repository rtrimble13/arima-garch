# Preset-driven thin Makefile wrapper for CMake (>= 3.14)
#
# This Makefile assumes you have a CMakePresets.json that defines:
#   configure presets: ninja-release, ninja-debug, ninja-relwithdebinfo
# and that each preset uses a predictable binaryDir like:
#   build/ninja-release, build/ninja-debug, build/ninja-relwithdebinfo
#
# Usage:
#   make                 # Release: configure+build
#   make debug           # Debug: configure+build
#   make relwithdebinfo  # RelWithDebInfo: configure+build
#   make test            # run tests for selected BUILD (default Release)
#   make clean           # remove build dir for selected BUILD
#   make reconfigure     # clean + configure
#
# Overrides:
#   make BUILD=Debug
#   make CONF_PRESET=my-preset BUILD_DIR=some/dir
#   make N=12

SHELL := /bin/sh

CMAKE ?= cmake
CTEST ?= ctest

# ---- Build selection ----
BUILD ?= Release

# Default preset names (overrideable)
ifeq ($(BUILD),Release)
  CONF_PRESET  ?= ninja-release
  BUILD_PRESET ?= ninja-release
  TEST_PRESET  ?= ninja-release
  BUILD_DIR    ?= build/ninja-release
else ifeq ($(BUILD),Debug)
  CONF_PRESET  ?= ninja-debug
  BUILD_PRESET ?= ninja-debug
  TEST_PRESET  ?= ninja-debug
  BUILD_DIR    ?= build/ninja-debug
else ifeq ($(BUILD),RelWithDebInfo)
  CONF_PRESET  ?= ninja-relwithdebinfo
  BUILD_PRESET ?= ninja-relwithdebinfo
  TEST_PRESET  ?= ninja-relwithdebinfo
  BUILD_DIR    ?= build/ninja-relwithdebinfo
else ifeq ($(BUILD),MinSizeRel)
  CONF_PRESET  ?= ninja-minsizerel
  BUILD_PRESET ?= ninja-minsizerel
  TEST_PRESET  ?= ninja-minsizerel
  BUILD_DIR    ?= build/ninja-minsizerel
else
  # Custom build type/preset naming convention fallback
  CONF_PRESET  ?= ninja-$(BUILD)
  BUILD_PRESET ?= ninja-$(BUILD)
  TEST_PRESET  ?= ninja-$(BUILD)
  BUILD_DIR    ?= build/ninja-$(BUILD)
endif

# Optional build parallelism
N ?=
ifneq ($(strip $(N)),)
  BUILD_PARALLEL = --parallel $(N)
else
  BUILD_PARALLEL =
endif

.PHONY: all release debug relwithdebinfo minsizerel configure build test clean reconfigure info

all: release

release:
	@$(MAKE) BUILD=Release configure build

debug:
	@$(MAKE) BUILD=Debug configure build

relwithdebinfo:
	@$(MAKE) BUILD=RelWithDebInfo configure build

minsizerel:
	@$(MAKE) BUILD=MinSizeRel configure build

configure:
	@echo "==> Configuring via preset: $(CONF_PRESET)"
	@$(CMAKE) --preset $(CONF_PRESET)

build:
	@echo "==> Building via preset: $(BUILD_PRESET)"
	@$(CMAKE) --build --preset $(BUILD_PRESET) $(BUILD_PARALLEL)

test:
	@echo "==> Testing via preset: $(TEST_PRESET)"
	@$(CTEST) --preset $(TEST_PRESET) --output-on-failure

clean:
	@echo "==> Removing build directory: $(BUILD_DIR)"
	@rm -rf "$(BUILD_DIR)"

reconfigure: clean configure

info:
	@echo "BUILD=$(BUILD)"
	@echo "CONF_PRESET=$(CONF_PRESET)"
	@echo "BUILD_PRESET=$(BUILD_PRESET)"
	@echo "TEST_PRESET=$(TEST_PRESET)"
	@echo "BUILD_DIR=$(BUILD_DIR)"
