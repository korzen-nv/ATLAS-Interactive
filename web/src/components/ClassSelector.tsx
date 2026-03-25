import { useCallback } from "react";
import { useAppStore } from "../state/store";
import type { ClientMessage } from "../api/types";

interface Props {
  send: (msg: ClientMessage) => void;
}

export function ClassSelector({ send }: Props) {
  const currentObject = useAppStore((s) => s.currentObject);
  const numObjects = useAppStore((s) => s.numObjects);
  const palette = useAppStore((s) => s.palette);

  const currentColor = palette.find((p) => p.id === currentObject)?.color ?? [128, 128, 128];

  const handleIdChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const id = parseInt(e.target.value);
      if (id >= 1 && id <= numObjects) {
        send({ type: "set_object", id });
      }
    },
    [send, numObjects]
  );

  const handleClassChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const id = parseInt(e.target.value);
      send({ type: "set_object", id });
    },
    [send]
  );

  return (
    <div className="class-selector">
      <div
        className="color-indicator"
        style={{ backgroundColor: `rgb(${currentColor.join(",")})` }}
      />
      <label>
        ID:
        <input
          type="number"
          min={1}
          max={numObjects}
          value={currentObject}
          onChange={handleIdChange}
          className="object-id-input"
        />
      </label>
      <label>
        Class:
        <select value={currentObject} onChange={handleClassChange} className="class-select">
          {palette.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
