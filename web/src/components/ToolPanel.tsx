import { useAppStore } from "../state/store";
import type { ClientMessage } from "../api/types";

interface Props {
  send: (msg: ClientMessage) => void;
}

export function ToolPanel({ send }: Props) {
  const propagating = useAppStore((s) => s.propagating);
  const playing = useAppStore((s) => s.playing);
  const polygonMode = useAppStore((s) => s.polygonMode);

  return (
    <div className="tool-panel">
      <div className="tool-group">
        <button
          className="btn"
          onClick={() => send({ type: "propagate", direction: "forward" })}
        >
          {propagating ? "Pause" : "Propagate Forward"}
        </button>
        <button
          className="btn"
          onClick={() => send({ type: "propagate", direction: "backward" })}
        >
          {propagating ? "Pause" : "Propagate Backward"}
        </button>
      </div>

      <div className="tool-group">
        <button className="btn" onClick={() => send({ type: "commit" })}>
          Commit to Memory
        </button>
        <button
          className="btn"
          onClick={() => send({ type: "play_video", playing: !playing })}
        >
          {playing ? "Stop Video" : "Play Video"}
        </button>
      </div>

      <div className="tool-group">
        <button className="btn btn-secondary" onClick={() => send({ type: "reset_frame" })}>
          Reset Frame
        </button>
        <button className="btn btn-secondary" onClick={() => send({ type: "reset_object" })}>
          Reset Object
        </button>
      </div>

      <div className="tool-group">
        <button
          className="btn btn-secondary"
          onClick={() => send({ type: "clear_memory", permanent: false })}
        >
          Reset Non-Perm Memory
        </button>
        <button
          className="btn btn-danger"
          onClick={() => send({ type: "clear_memory", permanent: true })}
        >
          Reset All Memory
        </button>
      </div>

      <div className="tool-info">
        {polygonMode ? "Polygon Mode (middle-click to toggle)" : "Click Mode (middle-click to toggle)"}
      </div>
    </div>
  );
}
