import { useEffect, useMemo, useState } from "react";
import "./index.css";
import { loadAppData, type AppData } from "./data/load";
import { Browser } from "./ui/Browser";
import { Explorer } from "./ui/Explorer";
import { WatchMode } from "./ui/WatchMode";
import { type ModelMode } from "./ui/model";

// TODO: switch to https://issaahmed.com once the custom domain is wired up
const PORTFOLIO_CASE_STUDY =
  "https://issaahmed-com.vercel.app/projects/nba-shot-selection";
const SOURCE_REPO = "https://github.com/issaa71/nba-rl-sim";

function Header() {
  return (
    <header className="appbar">
      <div className="shell appbar__inner">
        <a className="appbar__brand" href=".">
          NBA Shot Selection <b>·</b> RL
        </a>
        <a
          className="appbar__link"
          href={PORTFOLIO_CASE_STUDY}
          target="_blank"
          rel="noreferrer"
        >
          Case study ↗
        </a>
      </div>
    </header>
  );
}

function Footer({ data }: { data: AppData | null }) {
  const checkpoint =
    data?.dueling.checkpoint ??
    "results_v9/pbrs_lr1e4/dueling_dqn_weights.pth";
  return (
    <footer className="appfoot">
      <div className="shell appfoot__inner">
        <p className="appfoot__line">
          Model checkpoint <code>{checkpoint}</code> (Dueling DQN) ·{" "}
          <code>results_v9/pbrs_dqn_nodist/dqn_weights.pth</code> (DQN).
        </p>
        <p className="appfoot__line">
          Canonical eval: Dueling EPSA +0.273 on the held-out test split,
          n=127,353 decision points.
        </p>
        <p className="appfoot__line">
          Every number on this page is computed in your browser — the trained
          weights run client-side and reproduce the Python Q-values to within
          1e-4.{" "}
          <a href={PORTFOLIO_CASE_STUDY} target="_blank" rel="noreferrer">
            Read the case study ↗
          </a>{" "}
          ·{" "}
          <a href={SOURCE_REPO} target="_blank" rel="noreferrer">
            Source on GitHub ↗
          </a>
        </p>
      </div>
    </footer>
  );
}

function Intro() {
  return (
    <div className="hero">
      <p className="eyebrow">Reinforcement learning · Deep Q-networks</p>
      <h1>When should an NBA player shoot?</h1>
      <p className="hero__lead">
        Two agents — a Dueling DQN and a plain DQN — were trained on tracking
        data to value every shoot-or-pass decision. Browse 40 curated
        possessions from the held-out test set, replay each decision, and drag
        any player to see the recommendation change live.
      </p>
    </div>
  );
}

export default function App() {
  const [data, setData] = useState<AppData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [watching, setWatching] = useState(false);
  const [model, setModel] = useState<ModelMode>("dueling");

  useEffect(() => {
    let alive = true;
    loadAppData()
      .then((d) => {
        if (alive) setData(d);
      })
      .catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  const selected = useMemo(() => {
    if (!data || !selectedId) return null;
    return data.possessions.possessions.find((p) => p.id === selectedId) ?? null;
  }, [data, selectedId]);

  // keep scroll sane when switching views
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [selectedId, watching]);

  if (error) {
    return (
      <>
        <Header />
        <main className="shell">
          <div className="center-state">
            <p>Could not load demo data.</p>
            <p className="mono" style={{ fontSize: 12 }}>
              {error}
            </p>
          </div>
        </main>
      </>
    );
  }

  if (!data) {
    return (
      <>
        <Header />
        <main className="shell">
          <div className="center-state">
            <div className="spinner" />
            <p>Loading models &amp; possessions…</p>
          </div>
        </main>
      </>
    );
  }

  return (
    <>
      <Header />
      <main className="shell">
        {watching ? (
          <WatchMode
            data={data}
            model={model}
            onModelChange={setModel}
            onExit={() => setWatching(false)}
          />
        ) : selected ? (
          <Explorer
            key={selected.id}
            possession={selected}
            tracking={data.tracking.get(selected.id) ?? null}
            trackingFc={data.trackingFc.get(selected.id) ?? null}
            data={data}
            model={model}
            onModelChange={setModel}
            onBack={() => setSelectedId(null)}
          />
        ) : (
          <>
            <Intro />
            <Browser
              possessions={data.possessions.possessions}
              onOpen={setSelectedId}
              onWatch={() => setWatching(true)}
            />
          </>
        )}
      </main>
      <Footer data={data} />
    </>
  );
}
