import { useState, useCallback } from "react";
import { exportVideo, exportBinaryMasks } from "../api/rest";
import { useAppStore } from "../state/store";

export function ImportExportPanel() {
  const [status, setStatus] = useState("");
  const addConsoleLine = useAppStore((s) => s.addConsoleLine);

  const handleImportMask = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      await fetch("/api/import/mask", { method: "POST", body: form });
      addConsoleLine(`Mask imported: ${file.name}`);
    } catch {
      addConsoleLine("Failed to import mask");
    }
  }, [addConsoleLine]);

  const handleImportLayer = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      await fetch("/api/import/layer", { method: "POST", body: form });
      addConsoleLine(`Layer imported: ${file.name}`);
    } catch {
      addConsoleLine("Failed to import layer");
    }
  }, [addConsoleLine]);

  const handleExportVideo = useCallback(async () => {
    setStatus("Exporting video...");
    try {
      const result = await exportVideo();
      setStatus(`Exported to: ${result.path}`);
      addConsoleLine(`Video exported to ${result.path}`);
    } catch {
      setStatus("Export failed");
    }
  }, [addConsoleLine]);

  const handleExportMasks = useCallback(async () => {
    setStatus("Exporting masks...");
    try {
      const result = await exportBinaryMasks();
      setStatus(`Exported to: ${result.path}`);
      addConsoleLine(`Binary masks exported to ${result.path}`);
    } catch {
      setStatus("Export failed");
    }
  }, [addConsoleLine]);

  return (
    <div className="import-export-panel">
      <div className="tool-group">
        <label className="btn btn-secondary file-btn">
          Import Mask
          <input type="file" accept="image/*" onChange={handleImportMask} hidden />
        </label>
        <label className="btn btn-secondary file-btn">
          Import Layer
          <input type="file" accept="image/*" onChange={handleImportLayer} hidden />
        </label>
      </div>
      <div className="tool-group">
        <button className="btn btn-secondary" onClick={handleExportVideo}>
          Export Video
        </button>
        <button className="btn btn-secondary" onClick={handleExportMasks}>
          Export Binary Masks
        </button>
      </div>
      {status && <div className="export-status">{status}</div>}
    </div>
  );
}
