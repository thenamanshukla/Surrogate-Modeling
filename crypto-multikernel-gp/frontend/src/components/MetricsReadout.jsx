import AnimatedNumber from "./AnimatedNumber.jsx";

export default function MetricsReadout({ metrics }) {
  const items = [
    { key: "rmse", label: "RMSE", decimals: 4 },
    { key: "mae", label: "MAE", decimals: 4 },
    { key: "directional_accuracy", label: "DIRECTIONAL ACCURACY", decimals: 1, percent: true },
  ];

  return (
    <div className="bg-panel border border-gridline rounded-sm p-5">
      <h2 className="font-display text-sm tracking-wide text-text-primary mb-4">
        POINT FORECAST METRICS
      </h2>
      <div className="grid grid-cols-3 gap-4">
        {items.map(({ key, label, decimals, percent }) => {
          const raw = metrics ? metrics[key] : null;
          const value = raw !== null && raw !== undefined && percent ? raw * 100 : raw;
          return (
            <div
              key={key}
              className="bg-panel-raised rounded-sm p-4 transition-transform duration-200 hover:-translate-y-0.5 hover:border-signature border border-transparent"
            >
              <div className="text-text-secondary text-xs font-mono mb-1">{label}</div>
              <div className="font-mono text-2xl text-text-primary tabular-nums">
                <AnimatedNumber value={value} decimals={decimals} suffix={percent ? "%" : ""} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
