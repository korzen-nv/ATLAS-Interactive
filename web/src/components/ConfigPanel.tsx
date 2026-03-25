import { useState, useCallback } from "react";
import type { ClientMessage } from "../api/types";

interface Props {
  send: (msg: ClientMessage) => void;
}

export function ConfigPanel({ send }: Props) {
  const [workMemMin, setWorkMemMin] = useState(5);
  const [workMemMax, setWorkMemMax] = useState(10);
  const [longMemMax, setLongMemMax] = useState(10000);
  const [memEvery, setMemEvery] = useState(5);

  const applyConfig = useCallback(() => {
    send({
      type: "update_config",
      work_mem_min: workMemMin,
      work_mem_max: workMemMax,
      long_mem_max: longMemMax,
      mem_every: memEvery,
    });
  }, [send, workMemMin, workMemMax, longMemMax, memEvery]);

  return (
    <div className="config-panel">
      <h3>Memory Config</h3>
      <label>
        Min working memory frames:
        <input
          type="number"
          min={1}
          max={100}
          value={workMemMin}
          onChange={(e) => setWorkMemMin(parseInt(e.target.value))}
          onBlur={applyConfig}
        />
      </label>
      <label>
        Max working memory frames:
        <input
          type="number"
          min={2}
          max={100}
          value={workMemMax}
          onChange={(e) => setWorkMemMax(parseInt(e.target.value))}
          onBlur={applyConfig}
        />
      </label>
      <label>
        Max long-term memory:
        <input
          type="number"
          min={1000}
          max={100000}
          step={1000}
          value={longMemMax}
          onChange={(e) => setLongMemMax(parseInt(e.target.value))}
          onBlur={applyConfig}
        />
      </label>
      <label>
        Memory frame every (r):
        <input
          type="number"
          min={1}
          max={100}
          value={memEvery}
          onChange={(e) => setMemEvery(parseInt(e.target.value))}
          onBlur={applyConfig}
        />
      </label>
    </div>
  );
}
