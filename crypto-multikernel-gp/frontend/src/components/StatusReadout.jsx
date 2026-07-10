const STATUS_LABELS = {
  idle: "IDLE",
  queued: "QUEUED",
  fetching_data: "FETCHING DATA",
  engineering_features: "ENGINEERING FEATURES",
  training: "TRAINING",
  done: "DONE",
  error: "ERROR",
};

export default function StatusReadout({ status, windowsDone, windowsTotal, progress }) {
  const label = STATUS_LABELS[status] || status.toUpperCase();
  const isTraining = status === "training";
  const dotColor =
    status === "error" ? "bg-sentiment" : status === "done" ? "bg-fundamental" : "bg-signature";

  return (
    <div className="flex flex-col gap-1.5 font-mono text-xs min-w-[160px]">
      <div className="flex items-center gap-3">
        <span className={`w-2 h-2 rounded-full ${dotColor} ${isTraining ? "animate-pulse" : ""}`} />
        <span className="text-text-secondary tracking-wider transition-opacity duration-300">{label}</span>
        {isTraining && windowsTotal > 0 && (
          <span className="text-text-secondary">
            window {windowsDone} of {windowsTotal}
          </span>
        )}
      </div>
      {isTraining && windowsTotal > 0 && (
        <div className="w-40 h-1 bg-panel-raised rounded-full overflow-hidden">
          <div
            className="h-full bg-signature rounded-full transition-all duration-500 ease-out"
            style={{ width: `${Math.max(progress, 3)}%` }}
          />
        </div>
      )}
    </div>
  );
}
