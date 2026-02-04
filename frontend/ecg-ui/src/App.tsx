import { useEffect, useState } from "react";

/* ---------- Theme CSS ---------- */
const THEME_CSS = `
:root{--bg:#0f1724;--card:#0b1220;--muted:#9aa6b2;--accent:#06b6d4}
*{box-sizing:border-box}
body{
  margin:0;
  font-family:Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto;
  background:linear-gradient(180deg,#071021 0%, #0b1725 100%);
  color:#e6eef6
}
.app{max-width:900px;margin:40px auto;padding:20px}
.card{background:var(--card);padding:20px;border-radius:12px}
.btn{
  padding:10px 14px;
  border-radius:8px;
  border:none;
  background:var(--accent);
  color:#042028;
  font-weight:600;
  cursor:pointer
}
.btn:disabled{opacity:0.7;cursor:not-allowed}
.input{
  padding:10px 12px;
  border-radius:8px;
  border:1px solid rgba(255,255,255,0.1);
  background:rgba(255,255,255,0.02);
  color:#e6eef6;
  width:100%;
  margin-bottom:10px
}
.muted{color:var(--muted)}
.error{color:#f87171;margin-top:10px}
`;

/* ---------- Types ---------- */
type PredictionResult = {
  filename: string;
  prediction: string;
  confidence: number;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string;

function App() {
  /* Inject CSS */
  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = THEME_CSS;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  /* ---------- State (STRICT TYPES) ---------- */
  const [datFile, setDatFile] = useState<File | null>(null);
  const [heaFile, setHeaFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  /* ---------- Submit ---------- */
  const handleSubmit = async () => {
    if (!API_BASE_URL) {
      setError("Backend URL not configured");
      return;
    }

    if (!datFile || !heaFile) {
      setError("Please upload BOTH .dat and .hea files");
      return;
    }

    const formData = new FormData();
    formData.append("dat_file", datFile);
    formData.append("hea_file", heaFile);

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err: unknown) {
      setError("Failed to connect to backend or invalid ECG files");
    } finally {
      setLoading(false);
    }
  };

  /* ---------- UI ---------- */
  return (
    <div className="app">
      <h1>ECG Arrhythmia Detection System</h1>
      <p className="muted">
        Upload PTB-XL ECG <b>.dat</b> and <b>.hea</b> files
      </p>

      <div className="card" style={{ marginTop: 20 }}>
        <input
          type="file"
          accept=".dat"
          className="input"
          onChange={(e) => setDatFile(e.target.files?.[0] ?? null)}
        />

        <input
          type="file"
          accept=".hea"
          className="input"
          onChange={(e) => setHeaFile(e.target.files?.[0] ?? null)}
        />

        <button className="btn" onClick={handleSubmit} disabled={loading}>
          {loading ? "Analyzing..." : "Upload & Predict"}
        </button>

        {error && <div className="error">{error}</div>}
      </div>

      {result && (
        <div className="card" style={{ marginTop: 20 }}>
          <h3>Prediction Result</h3>
          <p><b>File:</b> {result.filename}</p>
          <p><b>Prediction:</b> {result.prediction}</p>
          <p>
            <b>Confidence:</b> {(result.confidence * 100).toFixed(2)}%
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
