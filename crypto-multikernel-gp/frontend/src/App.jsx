import { useState, useRef, useCallback } from "react";
import AssetSelector from "./components/AssetSelector.jsx";
import StatusReadout from "./components/StatusReadout.jsx";
import LiveTracePanel from "./components/LiveTracePanel.jsx";
import MetricsReadout from "./components/MetricsReadout.jsx";
import CalibrationTable from "./components/CalibrationTable.jsx";
import ModelSpecPanel from "./components/ModelSpecPanel.jsx";

const POLL_INTERVAL_MS = 1200;

export default function App() {
  const [asset, setAsset] = useState(null);
  const [status, setStatus] = useState("idle");
  const [records, setRecords] = useState([]);
  const [windowsDone, setWindowsDone] = useState(0);
  const [windowsTotal, setWindowsTotal] = useState(0);
  const [progress, setProgress] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [calibrationTable, setCalibrationTable] = useState(null);
  const [error, setError] = useState(null);

  const pollRef = useRef(null);
  const sinceIndexRef = useRef(0);

  const isBusy = status === "queued" || status === "fetching_data" || status === "engineering_features" || status === "training";

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const pollJob = useCallback((jobId) => {
    pollRef.current = setInterval(async () => {
      try {
        const resp = await fetch(`/api/jobs/${jobId}?since_index=${sinceIndexRef.current}`);
        if (!resp.ok) {
          throw new Error(`job polling failed with status ${resp.status}`);
        }
        const job = await resp.json();

        if (job.new_records && job.new_records.length > 0) {
          setRecords((prev) => [...prev, ...job.new_records]);
          sinceIndexRef.current = job.total_records;
        }

        setStatus(job.status);
        setWindowsDone(job.windows_done);
        setWindowsTotal(job.windows_total);
        setProgress(job.progress);

        if (job.status === "done") {
          setMetrics(job.point_metrics);
          setCalibrationTable(job.calibration_table);
          stopPolling();
        } else if (job.status === "error") {
          setError(job.error || "training failed for an unknown reason");
          stopPolling();
        }
      } catch (err) {
        setError(err.message);
        stopPolling();
      }
    }, POLL_INTERVAL_MS);
  }, [stopPolling]);

  const runTraining = useCallback(async () => {
    if (!asset) return;
    stopPolling();
    setError(null);
    setRecords([]);
    setMetrics(null);
    setCalibrationTable(null);
    setWindowsDone(0);
    setWindowsTotal(0);
    setProgress(0);
    sinceIndexRef.current = 0;
    setStatus("queued");

    try {
      const resp = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ asset }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || `request failed with status ${resp.status}`);
      }
      const { job_id } = await resp.json();
      pollJob(job_id);
    } catch (err) {
      setStatus("error");
      setError(err.message);
    }
  }, [asset, pollJob, stopPolling]);

  return (
    <div className="min-h-screen max-w-6xl mx-auto px-6 py-10">
      <header className="mb-10 border-b border-gridline pb-8">
        <h1 className="font-display text-2xl text-text-primary tracking-tight">
          MULTI KERNEL GP <span className="text-text-secondary">live uncertainty decomposition</span>
        </h1>
        <p className="font-mono text-xs text-text-secondary mt-3 max-w-2xl leading-relaxed">
          Runs the full walk forward fit live, in the background, streaming each window's
          predictions and per modality variance decomposition as it completes.
        </p>
      </header>

      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <AssetSelector selected={asset} onSelect={setAsset} disabled={isBusy} />
        <div className="flex items-center gap-4">
          <StatusReadout status={status} windowsDone={windowsDone} windowsTotal={windowsTotal} progress={progress} />
          <button
            onClick={runTraining}
            disabled={!asset || isBusy}
            className="font-mono text-xs px-5 py-2.5 rounded-sm bg-signature text-ink font-medium
              disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
          >
            {isBusy ? "training…" : "run live fit"}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-6 border border-sentiment/50 bg-sentiment/10 rounded-sm p-4 font-mono text-xs text-sentiment animate-fadein">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-5 mb-5 animate-fadein">
        <LiveTracePanel records={records} isLive={isBusy} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5 animate-fadein">
        <MetricsReadout metrics={metrics} />
        <ModelSpecPanel />
      </div>

      <div className="grid grid-cols-1 gap-5 mb-10 animate-fadein">
        <CalibrationTable calibrationTable={calibrationTable} />
      </div>

      <footer className="font-mono text-[11px] text-text-secondary border-t border-gridline pt-6 pb-10">
        Live fit on real CoinGecko market data and the alternative.me Fear and Greed Index.
        Free tier history capped at 365 days. Not financial advice. A research instrument, not a
        signal generator.
      </footer>
    </div>
  );
}
