# Repository Guidelines & Operational Rules

## 1. Upstream Protection Guardrail
- Under NO circumstances attempt to push to any upstream repository (`lingo-db/lingo-db` or any official repo).
- Push and branch exclusively on the personal fork (`ikabadzhov/lingo-db` or user personal fork).
- Verify remotes with `git remote -v` before any git operation.

## 2. Spec-Driven Development (SDD) & Sanity Gates
- Always execute changes in phased, linear order.
- Verify every phase with its explicit sanity gate test command (must exit with code 0).
- Create an atomic git commit on the fork feature branch after each passing gate.

