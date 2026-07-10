# SWE-Forge Benchmark Report

- Shipped tasks: **1**
- Overall: **PASS**

## Headline A - gold solvability
- gold = **100%** (1/1) -- PASS
- deterministic across reruns: True

## Headline B - frontier solve-rate
- stated threshold: **0.5000**
- measured frontier solve-rate: **0.1667** (< threshold and > 0)

## Per-model panel solve-rates

| model | tier | tasks | solves/k | solve-rate |
| --- | --- | --- | --- | --- |
| openai/gpt-4o-mini | weak | 1 | 0/6 | 0.0000 |
| anthropic/claude-sonnet-4-6 | mid | 1 | 0/6 | 0.0000 |
| anthropic/claude-opus-4-8 | frontier | 1 | 1/6 | 0.1667 |
| openai/gpt-5.5 | frontier | 1 | 1/6 | 0.1667 |

- per-tier (pooled): weak=0.0000, mid=0.0000, frontier=0.1667 (weak <= mid <= frontier: True; gold 100% >= frontier: True)

## IRT difficulty / discrimination
- difficulty: mean=1.3427, min=1.3427, max=1.3427
- discrimination: mean=4.7038, min=4.7038, max=4.7038

## Breakdown
- by generator: bug_combination=1
- by language: python=1
- breakdown sums to shipped total: True

## Counts reconciliation
- tasks/*/ = 1, jsonl = 1, parquet = 1 (reconciled: True)

## Provenance audit
- completeness: 1/1 complete (PASS)
- consistency: 1/1 consistent (PASS)
