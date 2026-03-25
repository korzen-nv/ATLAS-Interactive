import { useAppStore } from "../state/store";

export function MemoryGauges() {
  const mem = useAppStore((s) => s.memoryStatus);
  if (!mem) return null;

  return (
    <div className="memory-gauges">
      <Gauge
        label="Permanent Memory"
        value={mem.perm_tokens}
        max={mem.perm_tokens || 1}
        format={`${mem.perm_tokens}`}
      />
      <Gauge
        label="Working Memory"
        value={mem.work_tokens}
        max={mem.max_work_tokens}
        format={`${mem.work_tokens} / ${mem.max_work_tokens}`}
      />
      <Gauge
        label="Long-term Memory"
        value={mem.long_tokens}
        max={mem.max_long_tokens}
        format={`${mem.long_tokens} / ${mem.max_long_tokens}`}
      />
      {mem.gpu_total_gb > 0 && (
        <>
          <Gauge
            label="GPU Memory"
            value={mem.gpu_used_gb}
            max={mem.gpu_total_gb}
            format={`${mem.gpu_used_gb} GB / ${mem.gpu_total_gb} GB`}
          />
          <Gauge
            label="Torch Memory"
            value={mem.torch_used_gb}
            max={mem.gpu_total_gb}
            format={`${mem.torch_used_gb} GB`}
          />
        </>
      )}
    </div>
  );
}

function Gauge({
  label,
  value,
  max,
  format,
}: {
  label: string;
  value: number;
  max: number;
  format: string;
}) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="gauge">
      <div className="gauge-label">
        <span>{label}</span>
        <span className="gauge-value">{format}</span>
      </div>
      <div className="gauge-bar">
        <div className="gauge-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
