# AGENTS.md

## Repository Purpose

Teaching exercises for NOMICC 2026 (Summer School on Nonlinear Optimization with Mixed Integer and Complementarity Constraints). This is an educational repository, not a production codebase.

## Environment Setup

The virtual environment is already created at `venv/`. To activate:

```bash
source venv/bin/activate
```

If you need to re-create the environment from scratch, use the install scripts:
- Linux: `./install_linux.sh`
- macOS (aarch64): `./install_macos_aarch64.sh`

These scripts clone `nosnoc_py`, download `libMad` binaries, create the venv, and install all dependencies.

## Key Dependencies

- **CasADi** (>=3.8) — nonlinear optimization and algorithmic differentiation
- **CAMINO** (`caminopy`) — MINLP solver interface
- **libMad** — shared library for CCOpt.jl/MadNLP.jl (requires `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`)
- **nosnoc** — nonsmooth optimal control (installed from source, branch `v1.0.0-rc`)

## Verification

After setup, verify the installation works:

```bash
python test_install.py --ccopt --camino --nosnoc
```

## Exercises

Exercises live in `exercises/` as subdirectories. Each has a `description.md` with problem details. Run exercise scripts from within their directory (they use relative imports):

```bash
cd exercises/ell0-portfolio
python main.py --help
```

## Platform-Specific Notes

- **macOS**: System Integrity Protection (SIP) must be disabled for `DYLD_LIBRARY_PATH` to work, or use `otool` to set absolute paths for `libcasadi_nlpsol_ccopt.dylib`.
- **Windows**: No install script provided; follow README instructions manually. Avoid Microsoft Store Python.
