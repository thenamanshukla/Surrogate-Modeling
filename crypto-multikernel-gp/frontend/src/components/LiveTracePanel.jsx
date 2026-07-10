import { useState, useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const MODALITY_COLORS = {
  var_fundamental: "#4FA8A0",
  var_technical: "#E0A458",
  var_sentiment: "#C46B8A",
};

const MODALITY_LABELS = {
  var_fundamental: "fundamental",
  var_technical: "technical",
  var_sentiment: "sentiment",
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="bg-panel-raised border border-gridline rounded-sm p-3 font-mono text-xs shadow-lg">
      <div className="text-text-secondary mb-1">{label}</div>
      {payload.map((entry) => (
        <div key={entry.dataKey} className="flex justify-between gap-4" style={{ color: entry.color }}>
          <span>{MODALITY_LABELS[entry.dataKey] || entry.dataKey}</span>
          <span>{Number(entry.value).toExponential(2)}</span>
        </div>
      ))}
      <div className="text-text-secondary mt-1 pt-1 border-t border-gridline">click point for full detail</div>
    </div>
  );
}

function DayDetail({ record, onClose }) {
  if (!record) return null;
  const dof = 4;
  const scale = Math.sqrt(Math.max(record.pred_var, 0) * (dof - 2) / dof);
  const q95 = 2.776;
  const lower = record.pred_mean - q95 * scale;
  const upper = record.pred_mean + q95 * scale;

  return (
    <div className="mt-4 bg-panel-raised border border-gridline rounded-sm p-5 animate-fadein">
      <div className="flex items-center justify-between mb-4">
        <span className="font-mono text-xs text-text-secondary tracking-wide">{record.date}</span>
        <button
          onClick={onClose}
          className="font-mono text-xs text-text-secondary hover:text-text-primary transition-colors"
        >
          close
        </button>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <div className="text-text-secondary text-xs font-mono mb-1">actual</div>
          <div className="font-mono text-lg text-text-primary">{record.y_true.toFixed(4)}</div>
        </div>
        <div>
          <div className="text-text-secondary text-xs font-mono mb-1">predicted</div>
          <div className="font-mono text-lg text-text-primary">{record.pred_mean.toFixed(4)}</div>
        </div>
        <div className="col-span-2">
          <div className="text-text-secondary text-xs font-mono mb-1">95 percent interval</div>
          <div className="font-mono text-lg text-text-primary">
            [{lower.toFixed(4)}, {upper.toFixed(4)}]
          </div>
        </div>
      </div>
      <div className="text-text-secondary text-xs font-mono mb-2">variance contribution</div>
      <div className="space-y-2">
        {Object.entries(MODALITY_LABELS).map(([key, label]) => {
          const value = record[key];
          const total = record.var_fundamental + record.var_technical + record.var_sentiment;
          const share = total > 0 ? (value / total) * 100 : 0;
          return (
            <div key={key} className="flex items-center gap-3">
              <span className="font-mono text-xs text-text-secondary w-20">{label}</span>
              <div className="flex-1 h-2 bg-ink rounded-sm overflow-hidden">
                <div
                  className="h-full rounded-sm transition-all duration-500 ease-out"
                  style={{ width: `${share}%`, backgroundColor: MODALITY_COLORS[key] }}
                />
              </div>
              <span className="font-mono text-xs text-text-primary w-16 text-right">
                {share.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
      <div className="mt-3 font-mono text-xs text-interaction">
        interaction term {record.var_interaction.toFixed(4)}
      </div>
    </div>
  );
}

export default function LiveTracePanel({ records, isLive }) {
  const [selectedDate, setSelectedDate] = useState(null);
  const hasData = records && records.length > 0;

  const selectedRecord = useMemo(
    () => (selectedDate ? records.find((r) => r.date === selectedDate) : null),
    [selectedDate, records]
  );

  const handleChartClick = (state) => {
    if (state && state.activeLabel) {
      setSelectedDate(state.activeLabel);
    }
  };

  return (
    <div className="bg-panel border border-gridline rounded-sm p-5 relative overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <h2 className="font-display text-sm tracking-wide text-text-primary">
          PREDICTIVE VARIANCE PER MODALITY DECOMPOSITION
        </h2>
        <div className="flex gap-4 font-mono text-xs">
          {Object.entries(MODALITY_LABELS).map(([key, label]) => (
            <div key={key} className="flex items-center gap-1.5">
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: MODALITY_COLORS[key] }}
              />
              <span className="text-text-secondary">{label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="h-80 cursor-pointer">
        {!hasData ? (
          <div className="h-full flex items-center justify-center font-mono text-xs text-text-secondary">
            select an asset and run live training to populate this trace
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={records}
              margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
              onClick={handleChartClick}
            >
              <CartesianGrid stroke="#232F38" strokeDasharray="2 4" />
              <XAxis
                dataKey="date"
                tick={{ fill: "#8A9BA5", fontSize: 10, fontFamily: "IBM Plex Mono" }}
                minTickGap={40}
              />
              <YAxis
                tick={{ fill: "#8A9BA5", fontSize: 10, fontFamily: "IBM Plex Mono" }}
                width={50}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="var_fundamental"
                stackId="1"
                stroke={MODALITY_COLORS.var_fundamental}
                fill={MODALITY_COLORS.var_fundamental}
                fillOpacity={0.65}
                isAnimationActive={isLive}
                animationDuration={400}
                animationEasing="ease-out"
                activeDot={{ r: 4, cursor: "pointer" }}
              />
              <Area
                type="monotone"
                dataKey="var_technical"
                stackId="1"
                stroke={MODALITY_COLORS.var_technical}
                fill={MODALITY_COLORS.var_technical}
                fillOpacity={0.65}
                isAnimationActive={isLive}
                animationDuration={400}
                animationEasing="ease-out"
                activeDot={{ r: 4, cursor: "pointer" }}
              />
              <Area
                type="monotone"
                dataKey="var_sentiment"
                stackId="1"
                stroke={MODALITY_COLORS.var_sentiment}
                fill={MODALITY_COLORS.var_sentiment}
                fillOpacity={0.65}
                isAnimationActive={isLive}
                animationDuration={400}
                animationEasing="ease-out"
                activeDot={{ r: 4, cursor: "pointer" }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
      {isLive && (
        <div className="absolute top-5 right-5 flex items-center gap-2 font-mono text-xs text-signature">
          <span className="w-1.5 h-1.5 rounded-full bg-signature animate-ping" />
          live
        </div>
      )}
      {selectedRecord && <DayDetail record={selectedRecord} onClose={() => setSelectedDate(null)} />}
    </div>
  );
}
