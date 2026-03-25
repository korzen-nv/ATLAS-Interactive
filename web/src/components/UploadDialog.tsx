import { useState, useCallback } from "react";
import { uploadVideo, uploadImages, createSession } from "../api/rest";
import { useAppStore } from "../state/store";

export function UploadDialog() {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [workspace, setWorkspace] = useState("");
  const setSessionId = useAppStore((s) => s.setSessionId);
  const setNumObjects = useAppStore((s) => s.setNumObjects);
  const setImageDimensions = useAppStore((s) => s.setImageDimensions);

  const handleSessionCreated = useCallback(
    (info: { session_id: string; width: number; height: number; num_objects?: number; total_frames: number }) => {
      setSessionId(info.session_id);
      setImageDimensions(info.width, info.height);
      if (info.num_objects) setNumObjects(info.num_objects);
    },
    [setSessionId, setImageDimensions, setNumObjects]
  );

  const handleFileUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setUploading(true);
      setError(null);
      try {
        const isZip = file.name.endsWith(".zip");
        const info = isZip ? await uploadImages(file) : await uploadVideo(file);
        handleSessionCreated(info);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [handleSessionCreated]
  );

  const handleWorkspaceOpen = useCallback(async () => {
    if (!workspace.trim()) return;
    setUploading(true);
    setError(null);
    try {
      const info = await createSession({ workspace: workspace.trim() });
      handleSessionCreated(info);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to open workspace");
    } finally {
      setUploading(false);
    }
  }, [workspace, handleSessionCreated]);

  return (
    <div className="upload-dialog">
      <div className="upload-card">
        <h1>ATLAS-Interactive</h1>
        <p>Interactive video labeling for surgical segmentation</p>

        <div className="upload-section">
          <h3>Upload Video or Images</h3>
          <label className="file-upload-label">
            <input
              type="file"
              accept="video/*,.zip"
              onChange={handleFileUpload}
              disabled={uploading}
            />
            {uploading ? "Uploading & processing..." : "Choose video file or .zip of images"}
          </label>
        </div>

        <div className="upload-divider">or</div>

        <div className="upload-section">
          <h3>Open Existing Workspace</h3>
          <div className="workspace-input-row">
            <input
              type="text"
              placeholder="./workspace/my_video.mp4"
              value={workspace}
              onChange={(e) => setWorkspace(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleWorkspaceOpen()}
            />
            <button className="btn" onClick={handleWorkspaceOpen} disabled={uploading}>
              Open
            </button>
          </div>
        </div>

        {error && <div className="upload-error">{error}</div>}
      </div>
    </div>
  );
}
