import { useEffect } from "react";
import type { ClientMessage } from "../api/types";

export function useKeyboardShortcuts(
  send: (msg: ClientMessage) => void,
  numObjects: number
) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      // Number keys 1-9 for object selection
      const num = parseInt(e.key);
      if (num >= 1 && num <= Math.min(9, numObjects)) {
        e.preventDefault();
        send({ type: "set_object", id: num });
        return;
      }

      switch (e.key) {
        case "ArrowLeft":
          e.preventDefault();
          if (e.altKey) {
            send({ type: "navigate", frame: 0 });
          } else {
            send({ type: "navigate", frame: -999 }); // handled as relative in special way
            // Actually, we need to send specific frame. Use a workaround:
            // We'll send a navigate with a special step
          }
          break;
        case "ArrowRight":
          e.preventDefault();
          break;
        case "c":
        case "C":
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            send({ type: "commit" });
          }
          break;
        case "f":
        case "F":
        case " ":
          e.preventDefault();
          send({ type: "propagate", direction: "forward" });
          break;
        case "b":
        case "B":
          e.preventDefault();
          send({ type: "propagate", direction: "backward" });
          break;
        case "t":
        case "T":
          e.preventDefault();
          send({ type: "toggle_vis_mode" });
          break;
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [send, numObjects]);
}

/**
 * Separate hook for arrow key navigation that needs current frame state.
 */
export function useArrowNavigation(
  send: (msg: ClientMessage) => void,
  currentFrame: number,
  totalFrames: number
) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        const step = e.altKey ? currentFrame : e.shiftKey ? 10 : 1;
        send({ type: "navigate", frame: Math.max(0, currentFrame - step) });
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        const step = e.altKey ? totalFrames - 1 - currentFrame : e.shiftKey ? 10 : 1;
        send({ type: "navigate", frame: Math.min(totalFrames - 1, currentFrame + step) });
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [send, currentFrame, totalFrames]);
}
