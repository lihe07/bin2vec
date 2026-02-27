# bin2vec
Dataset collector for bin2vec model

## Usage

uv sync
uv run bin2vec --help

Compatible with podman with the following command:

export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock


## Model

Bin2Vec fingerprints a function according to disasm and CFG.

Goals:

- Different optimization levels should have the same fingerprint. So printf with -O0 and printf with -O2 should have the same fingerprint.
- Different compilers should have the same fingerprint. 

Not goals:

- The fingerprint don't need to share across ISAs. 

  printf on Arm64 can have different fingerprint with x86-64. 

- Different calling convention can have different fingerprint. 
  
  printf with System V calling convention can have different fingerprint with printf with Microsoft calling convention.

  But the dataset MUST cover enough calling conventions and ISAs, so that the model can learn to ignore them.
