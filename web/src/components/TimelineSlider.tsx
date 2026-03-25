import { useCallback } from "react";
import { useAppStore } from "../state/store";
import type { ClientMessage } from "../api/types";

interface Props {
  send: (msg: ClientMessage) => void;
}

export function TimelineSlider({ send }: Props) {
  const currentFrame = useAppStore((s) => s.currentFrame);
  const totalFrames = useAppStore((s) => s.totalFrames);
  const frameName = useAppStore((s) => s.frameName);
  const progress = useAppStore((s) => s.progress);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const frame = parseInt(e.target.value);
      send({ type: "navigate", frame });
    },
    [send]
  );

  return (
    <div className="timeline-container">
      <input
        type="range"
        className="timeline-slider"
        min={0}
        max={Math.max(0, totalFrames - 1)}
        value={currentFrame}
        onChange={handleChange}
      />
      <div className="timeline-info">
        <span className="frame-counter">
          {currentFrame} / {totalFrames - 1}
        </span>
        <span className="frame-name">{frameName}</span>
        {progress > 0 && progress < 1 && (
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
          </div>
        )}
      </div>
    </div>
  );
}
