# globalopt — Lean proofs for TutorialGO

Machine-checked facts behind `../answers.tex`. All the proofs are in `Globalopt/Basic.lean`.

```bash
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --no-modify-path
export PATH=$HOME/.elan/bin:$PATH
lake exe cache get   # first time only: downloads the prebuilt Mathlib (~8 GB in .lake/)
lake build           # ~45 s
```

`.lake/` is git-ignored.
