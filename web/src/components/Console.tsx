import { useRef, useEffect } from "react";
import { useAppStore } from "../state/store";

export function Console() {
  const lines = useAppStore((s) => s.consoleLines);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  return (
    <div className="console">
      <div className="console-content">
        {lines.map((line, i) => (
          <div key={i} className="console-line">
            {line}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
