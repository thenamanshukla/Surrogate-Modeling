export default function ModelSpecPanel() {
  return (
    <div className="bg-panel border border-gridline rounded-sm p-5 transition-colors duration-200 hover:border-text-secondary">
      <h2 className="font-display text-sm tracking-wide text-text-primary mb-4">MODEL SPEC</h2>
      <div className="font-mono text-xs text-text-secondary leading-relaxed space-y-3">
        <div className="text-text-primary">
          k(x, x') = k_fundamental(x_f, x_f') + k_technical(x_t, x_t') + k_sentiment(x_s, x_s')
        </div>
        <p>
          Each modality is passed through its own MLP encoder into a Matern 5/2 kernel. Because the
          kernel is additive, posterior predictive variance decomposes exactly into per modality
          diagonal terms plus a signed interaction residual. No approximation.
        </p>
        <p>
          Likelihood: Student t, degrees of freedom fixed at 4, fit via variational inference
          using GPyTorch and SVI, with held out early stopping per walk forward window.
        </p>
        <p className="text-interaction">
          Data: CoinGecko price, volume, market cap, supply, and developer activity, plus the
          alternative.me Crypto Fear and Greed Index. Free tier history capped at 365 days.
        </p>
      </div>
    </div>
  );
}
