const MODALITY_COLORS = {
  fundamental: "#4FA8A0",
  technical: "#E0A458",
  sentiment: "#C46B8A",
};

export default function CalibrationTable({ calibrationTable }) {
  if (!calibrationTable || calibrationTable.length === 0) {
    return (
      <div className="bg-panel border border-gridline rounded-sm p-5">
        <h2 className="font-display text-sm tracking-wide text-text-primary mb-4">
          CALIBRATION AND EMPIRICAL COVERAGE
        </h2>
        <div className="font-mono text-xs text-text-secondary py-8 text-center">
          awaiting completed run
        </div>
      </div>
    );
  }

  const columns = Object.keys(calibrationTable[0]).filter((c) => c !== "nominal_level");

  return (
    <div className="bg-panel border border-gridline rounded-sm p-5">
      <h2 className="font-display text-sm tracking-wide text-text-primary mb-4">
        CALIBRATION AND EMPIRICAL COVERAGE
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-xs">
          <thead>
            <tr className="border-b border-gridline text-text-secondary">
              <th className="text-left py-2 pr-4">nominal</th>
              {columns.map((col) => {
                const modality = col.replace("coverage_when_", "").replace("_dominant", "");
                const color = MODALITY_COLORS[modality];
                return (
                  <th key={col} className="text-left py-2 pr-4">
                    <div className="flex items-center gap-1.5">
                      {color && <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />}
                      <span>{col.replace("coverage_when_", "").replace("_dominant", "").replace("empirical_coverage_overall", "overall")}</span>
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {calibrationTable.map((row) => (
              <tr
                key={row.nominal_level}
                className="border-b border-gridline/50 transition-colors duration-150 hover:bg-panel-raised"
              >
                <td className="py-2 pr-4 text-text-primary">{(row.nominal_level * 100).toFixed(0)}%</td>
                {columns.map((col) => {
                  const value = row[col];
                  const diff = value !== undefined ? Math.abs(value - row.nominal_level) : null;
                  const flagColor = diff !== null && diff > 0.15 ? "text-sentiment" : "text-text-primary";
                  return (
                    <td key={col} className={`py-2 pr-4 ${flagColor} transition-colors duration-150`}>
                      {value !== undefined ? (value * 100).toFixed(1) + "%" : "n a"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
