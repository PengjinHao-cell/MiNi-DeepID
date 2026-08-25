# GitHub README and Wiki Design

## Goal

Present Mini-DeepID as an English-first, bilingual educational reproduction of the core ideas in the CUHK DeepID paper. The homepage must communicate the experimental boundary, method, protocol, measured result, and full report without claiming reproduction of the paper's 97.45% LFW verification benchmark.

## Structure

The root README contains an English overview, results, Mermaid pipeline, figures, quick start, original-versus-Mini comparison, repository map, citation, Chinese summary, and ethics statement. Version-controlled Wiki sources live in `wiki/`: Home, Methodology, Reproduction, Results and Analysis, and Limitations and Future Work.

Selected figures are copied to `docs/assets/` because local outputs are ignored by Git. Wiki pages use raw main-repository asset URLs.

## Acceptance

All links and images resolve, no placeholders remain, metrics match the frozen result, tests pass, and both pushes succeed without force-push.
