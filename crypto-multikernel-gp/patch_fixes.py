import os

def patch(path, old, new, label):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if new in content:
        print(f"skip (already applied): {label}")
        return
    if old not in content:
        print(f"WARNING could not find target text for: {label}")
        return
    content = content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"applied: {label}")


patch(
    "src/train.py",
    '''        pred_mean = reduce_to_per_point(mean_raw, n_test) * y_std.item() + y_mean.item()
        pred_var = reduce_to_per_point(var_raw, n_test) * (y_std.item() ** 2)''',
    '''        pred_mean = reduce_to_per_point(mean_raw, n_test) * y_std.item() + y_mean.item()
        pred_var = reduce_to_per_point(var_raw, n_test) * (y_std.item() ** 2)
        pred_var = np.clip(pred_var, 1e-8, None)
        pred_mean = np.where(np.isfinite(pred_mean), pred_mean, 0.0)
        pred_var = np.where(np.isfinite(pred_var), pred_var, 1e-8)''',
    "train.py clip pred_var and guard non finite values",
)

patch(
    "src/train.py",
    '''        window_records = []
        for i in range(len(test_y)):
            window_records.append(
                {
                    "asset": symbol,
                    "date": test_X_raw.index[i],
                    "y_true": test_y[i].item(),
                    "pred_mean": float(np.asarray(pred_mean[i]).reshape(-1)[0]),
                    "pred_var": float(np.asarray(pred_var[i]).reshape(-1)[0]),
                    "var_fundamental": float(np.asarray(decomp["diagonal"]["fundamental"][i]).reshape(-1)[0]),
                    "var_technical": float(np.asarray(decomp["diagonal"]["technical"][i]).reshape(-1)[0]),
                    "var_sentiment": float(np.asarray(decomp["diagonal"]["sentiment"][i]).reshape(-1)[0]),
                    "var_interaction": float(np.asarray(decomp["interaction"][i]).reshape(-1)[0]),
                    "dominant_modality": dominant[i],
                }
            )''',
    '''        def safe_float(x):
            v = float(np.asarray(x).reshape(-1)[0])
            return v if np.isfinite(v) else 0.0

        window_records = []
        for i in range(len(test_y)):
            window_records.append(
                {
                    "asset": symbol,
                    "date": test_X_raw.index[i],
                    "y_true": safe_float(test_y[i].item()),
                    "pred_mean": safe_float(pred_mean[i]),
                    "pred_var": max(safe_float(pred_var[i]), 1e-8),
                    "var_fundamental": max(safe_float(decomp["diagonal"]["fundamental"][i]), 0.0),
                    "var_technical": max(safe_float(decomp["diagonal"]["technical"][i]), 0.0),
                    "var_sentiment": max(safe_float(decomp["diagonal"]["sentiment"][i]), 0.0),
                    "var_interaction": safe_float(decomp["interaction"][i]),
                    "dominant_modality": dominant[i],
                }
            )''',
    "train.py sanitize window record floats",
)

patch(
    "backend/jobs.py",
    '''            if len(df) > 0:
                metrics = point_metrics(df)
                calib = calibration_table(df)
                with job.lock:
                    job.point_metrics = {k: float(v) for k, v in metrics.items()}
                    job.calibration_table = calib.to_dict(orient="records")''',
    '''            if len(df) > 0:
                metrics = point_metrics(df)
                calib = calibration_table(df)

                def safe(v):
                    v = float(v)
                    return v if (v == v and v not in (float("inf"), float("-inf"))) else 0.0

                with job.lock:
                    job.point_metrics = {k: safe(v) for k, v in metrics.items()}
                    calib_records = calib.to_dict(orient="records")
                    for row in calib_records:
                        for k, v in row.items():
                            if isinstance(v, float):
                                row[k] = safe(v)
                    job.calibration_table = calib_records''',
    "jobs.py sanitize final metrics and calibration table",
)

patch(
    "backend/jobs.py",
    '''        except Exception as e:
            with job.lock:
                job.status = JobStatus.ERROR
                job.error = f"{str(e)}\\n{traceback.format_exc()}"''',
    '''        except Exception as e:
            tb = traceback.format_exc()
            print(f"job {job.job_id} failed for asset {job.asset}:")
            print(tb)
            with job.lock:
                job.status = JobStatus.ERROR
                job.error = f"{str(e)}\\n{tb}"''',
    "jobs.py print traceback to console on failure",
)

patch(
    "frontend/src/main.jsx",
    '''import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);''',
    '''import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import ErrorBoundary from "./ErrorBoundary.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);''',
    "main.jsx wrap App with ErrorBoundary",
)

error_boundary_path = "frontend/src/ErrorBoundary.jsx"
if not os.path.exists(error_boundary_path):
    with open(error_boundary_path, "w", encoding="utf-8") as f:
        f.write('''import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error && error.message ? error.message : "unknown error" };
  }

  componentDidCatch(error, info) {
    console.error("render error caught by boundary", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center px-6">
          <div className="max-w-lg border border-sentiment/50 bg-sentiment/10 rounded-sm p-6 font-mono text-sm text-sentiment">
            <div className="text-text-primary font-display text-base mb-2">something broke while rendering</div>
            <div className="text-xs leading-relaxed">{this.state.message}</div>
            <div className="text-text-secondary text-xs mt-4">
              reload the page to try again. if this keeps happening, check the browser console
              for the full stack trace and share it.
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
''')
    print("created: frontend/src/ErrorBoundary.jsx")
else:
    print("skip (already exists): frontend/src/ErrorBoundary.jsx")

print("done")
