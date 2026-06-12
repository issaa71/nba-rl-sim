# NBA Shot Selection — RL Possession Explorer

**Live: https://nba-rl-sim.vercel.app**

An interactive explorer for an offline-RL agent that decides *shoot or pass* on real NBA
possessions (2015–16 SportVU tracking data). The trained network runs **entirely in your
browser** — no server, no API. Pick a possession, watch the agent follow the play in real time,
then drag a defender and watch it change its mind.

Live companion to the case study at
[issaahmed-com.vercel.app/projects/nba-shot-selection](https://issaahmed-com.vercel.app/projects/nba-shot-selection).

## What you can do

- **Browse 40 curated possessions** from held-out test games, grouped by story:
  *Declined the shot* (the player shot; the agent saw a better pass), *Wanted the shot*
  (the player passed up a look the agent liked), and *Agreement* controls.
- **Watch the live agent**: playback interpolates the 2 Hz tracking samples to ~30 fps and
  re-runs the network every frame — the SHOOT/PASS call, target arrow, and Q-value bars follow
  the play continuously. Decision-point markers stay clickable for exact stepped inspection.
- **What-if mode**: pause anywhere, drag any player or defender, and the full pipeline —
  re-sort entities → zone-FG lookup → 73-feature state builder → network forward pass —
  recomputes live (~50 µs per evaluation).
- **Model toggle**: Dueling DQN (per-entity Deep Sets + type-aware advantage normalization),
  plain DQN, or the player's actual choices.

## Why you can trust the numbers

- The TypeScript inference engine and feature builder are tested against **250 golden vectors**
  exported from the original PyTorch/Python pipeline: features match to ≤1e-6 and Q-values to
  ≤1e-4 (measured worst case ≈1.6e-6). Run the parity suite with `npm test`.
- The footer's headline — **Dueling DQN EPSA +0.273 vs +0.044 for NBA players' actual
  choices** — comes from a deterministic greedy evaluation over all 127,353 decision points in
  the held-out test games (shot-quality EPSA: mean expected points of chosen shots minus the
  0.375 league baseline).
- Every Q-value shown is computed client-side from the shipped weights (~16K parameters).

## Stack

Vite · React · TypeScript · HTML5 Canvas. No UI framework, no chart library, no backend.
Total payload ≈ 1.6 MB including both models' weights and all possession data.

## Run it locally

```bash
npm install
npm test        # golden-vector parity gate (250 vectors, both models)
npm run dev
```

## Provenance

The models were trained offline (Dueling DQN with potential-based reward shaping) on 116,928
possessions segmented from 2015–16 SportVU tracking logs as part of a Western University AI
course project, then exported to JSON for client-side inference. Player names come from the NBA
Stats static index; zone-FG% tables are computed from public shot data.

Built by [Issa Ahmed](https://issaahmed-com.vercel.app).
