import { useEffect, useRef, useCallback } from "react";
import { WebSocketManager } from "../api/websocket";
import { useAppStore } from "../state/store";
import type { ClientMessage, ServerMessage } from "../api/types";

export function useWebSocket(sessionId: string | null) {
  const wsRef = useRef<WebSocketManager | null>(null);
  const store = useAppStore();

  const handleMessage = useCallback(
    (msg: ServerMessage) => {
      switch (msg.type) {
        case "frame":
          store.setFrameImage(msg.ti, msg.image, msg.name, msg.total_frames);
          break;
        case "state":
          store.updateFromState(msg as unknown as Record<string, unknown>);
          break;
        case "console":
          store.addConsoleLine(msg.text);
          break;
        case "progress":
          store.setProgress(msg.value);
          break;
        case "propagation_state":
          store.setPropagation(msg.propagating, msg.direction);
          break;
        case "memory_status":
          store.setMemoryStatus(msg);
          break;
        case "polygon_update":
          store.setPolygon(msg.polygon_mode, msg.points, msg.hover_first);
          break;
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  useEffect(() => {
    if (!sessionId) return;

    const ws = new WebSocketManager(sessionId);
    wsRef.current = ws;
    ws.onMessage(handleMessage);
    ws.connect();
    store.setConnected(true);

    return () => {
      ws.disconnect();
      wsRef.current = null;
      store.setConnected(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const send = useCallback((msg: ClientMessage) => {
    wsRef.current?.send(msg);
  }, []);

  return { send };
}
