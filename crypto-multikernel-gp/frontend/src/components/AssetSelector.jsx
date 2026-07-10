const ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"];

export default function AssetSelector({ selected, onSelect, disabled }) {
  return (
    <div className="flex flex-wrap gap-2">
      {ASSETS.map((asset) => {
        const isActive = asset === selected;
        return (
          <button
            key={asset}
            disabled={disabled}
            onClick={() => onSelect(asset)}
            className={`font-mono text-sm px-4 py-2 rounded-sm border transition-all duration-150
              ${
                isActive
                  ? "bg-signature text-ink border-signature"
                  : "bg-panel-raised text-text-secondary border-gridline hover:border-text-secondary hover:-translate-y-0.5"
              }
              ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer"}
            `}
          >
            {asset}
          </button>
        );
      })}
    </div>
  );
}
