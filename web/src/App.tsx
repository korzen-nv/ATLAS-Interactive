import { useEffect } from "react";
import { useAppStore } from "./state/store";
import { useWebSocket } from "./hooks/useWebSocket";
import { useKeyboardShortcuts, useArrowNavigation } from "./hooks/useKeyboardShortcuts";
import { getPalette } from "./api/rest";
import { CanvasView } from "./components/CanvasView";
import { TimelineSlider } from "./components/TimelineSlider";
import { ToolPanel } from "./components/ToolPanel";
import { ClassSelector } from "./components/ClassSelector";
import { VisModePicker } from "./components/VisModePicker";
import { MemoryGauges } from "./components/MemoryGauges";
import { ConfigPanel } from "./components/ConfigPanel";
import { Console } from "./components/Console";
import { UploadDialog } from "./components/UploadDialog";
import { ImportExportPanel } from "./components/ImportExportPanel";
import "./styles/globals.css";

function App() {
  const sessionId = useAppStore((s) => s.sessionId);
  const currentFrame = useAppStore((s) => s.currentFrame);
  const totalFrames = useAppStore((s) => s.totalFrames);
  const numObjects = useAppStore((s) => s.numObjects);
  const setPalette = useAppStore((s) => s.setPalette);

  const { send } = useWebSocket(sessionId);

  useKeyboardShortcuts(send, numObjects);
  useArrowNavigation(send, currentFrame, totalFrames);

  // Load palette on mount
  useEffect(() => {
    getPalette().then(setPalette).catch(console.error);
  }, [setPalette]);

  // Periodically request memory status
  useEffect(() => {
    if (!sessionId) return;
    const interval = setInterval(() => {
      send({ type: "get_memory_status" });
    }, 2000);
    return () => clearInterval(interval);
  }, [sessionId, send]);

  if (!sessionId) {
    return <UploadDialog />;
  }

  return (
    <div className="app-layout">
      <div className="main-area">
        <div className="canvas-area">
          <CanvasView send={send} />
        </div>
        <TimelineSlider send={send} />
        <div className="controls-bar">
          <ClassSelector send={send} />
          <VisModePicker send={send} />
          <ToolPanel send={send} />
        </div>
      </div>
      <div className="sidebar">
        <MemoryGauges />
        <ConfigPanel send={send} />
        <ImportExportPanel />
        <Console />
      </div>
    </div>
  );
}

export default App;
