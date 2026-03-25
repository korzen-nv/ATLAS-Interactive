import { useCallback } from "react";
import { useAppStore } from "../state/store";
import type { ClientMessage } from "../api/types";

const VIS_MODES = ["mask", "davis", "fade", "light", "popup", "rgba"];

interface Props {
  send: (msg: ClientMessage) => void;
}

export function VisModePicker({ send }: Props) {
  const visMode = useAppStore((s) => s.visMode);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      send({ type: "set_vis_mode", mode: e.target.value });
    },
    [send]
  );

  return (
    <label className="vis-mode-picker">
      Visualization:
      <select value={visMode} onChange={handleChange}>
        {VIS_MODES.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
    </label>
  );
}
