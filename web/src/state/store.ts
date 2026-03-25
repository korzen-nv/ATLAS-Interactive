import { create } from "zustand";
import type { PaletteEntry, MemoryStatus } from "../api/types";

interface AppState {
  // Session
  sessionId: string | null;
  connected: boolean;

  // Frame state
  currentFrame: number;
  totalFrames: number;
  frameImage: string | null; // base64 JPEG
  frameName: string;
  imageWidth: number;
  imageHeight: number;

  // Object selection
  currentObject: number;
  numObjects: number;

  // Visualization
  visMode: string;

  // Propagation
  propagating: boolean;
  propagateDirection: string;
  progress: number;

  // Polygon mode
  polygonMode: boolean;
  polygonPoints: [number, number][];
  hoverFirst: boolean;

  // Playback
  playing: boolean;

  // Memory
  memoryStatus: MemoryStatus | null;

  // Palette
  palette: PaletteEntry[];

  // Console
  consoleLines: string[];

  // Actions
  setSessionId: (id: string | null) => void;
  setConnected: (c: boolean) => void;
  setFrameImage: (ti: number, image: string, name: string, total: number) => void;
  setCurrentFrame: (ti: number) => void;
  setCurrentObject: (id: number) => void;
  setVisMode: (mode: string) => void;
  setPropagation: (propagating: boolean, direction: string) => void;
  setProgress: (value: number) => void;
  setPolygon: (mode: boolean, points: [number, number][], hover: boolean) => void;
  setPlaying: (playing: boolean) => void;
  setMemoryStatus: (status: MemoryStatus) => void;
  setPalette: (palette: PaletteEntry[]) => void;
  addConsoleLine: (text: string) => void;
  setImageDimensions: (w: number, h: number) => void;
  setNumObjects: (n: number) => void;
  updateFromState: (state: Record<string, unknown>) => void;
}

export const useAppStore = create<AppState>((set) => ({
  sessionId: null,
  connected: false,
  currentFrame: 0,
  totalFrames: 0,
  frameImage: null,
  frameName: "",
  imageWidth: 0,
  imageHeight: 0,
  currentObject: 1,
  numObjects: 3,
  visMode: "davis",
  propagating: false,
  propagateDirection: "none",
  progress: 0,
  polygonMode: false,
  polygonPoints: [],
  hoverFirst: false,
  playing: false,
  memoryStatus: null,
  palette: [],
  consoleLines: [],

  setSessionId: (id) => set({ sessionId: id }),
  setConnected: (c) => set({ connected: c }),
  setFrameImage: (ti, image, name, total) =>
    set({ currentFrame: ti, frameImage: image, frameName: name, totalFrames: total }),
  setCurrentFrame: (ti) => set({ currentFrame: ti }),
  setCurrentObject: (id) => set({ currentObject: id }),
  setVisMode: (mode) => set({ visMode: mode }),
  setPropagation: (propagating, direction) => set({ propagating, propagateDirection: direction }),
  setProgress: (value) => set({ progress: value }),
  setPolygon: (mode, points, hover) =>
    set({ polygonMode: mode, polygonPoints: points, hoverFirst: hover }),
  setPlaying: (playing) => set({ playing }),
  setMemoryStatus: (status) => set({ memoryStatus: status }),
  setPalette: (palette) => set({ palette }),
  addConsoleLine: (text) =>
    set((state) => ({
      consoleLines: [...state.consoleLines.slice(-99), text],
    })),
  setImageDimensions: (w, h) => set({ imageWidth: w, imageHeight: h }),
  setNumObjects: (n) => set({ numObjects: n }),
  updateFromState: (s) =>
    set({
      currentFrame: s.current_frame as number,
      totalFrames: s.total_frames as number,
      currentObject: s.current_object as number,
      visMode: s.vis_mode as string,
      propagating: s.propagating as boolean,
      propagateDirection: s.propagate_direction as string,
      polygonMode: s.polygon_mode as boolean,
      polygonPoints: s.polygon_points as [number, number][],
      playing: s.playing as boolean,
      frameName: s.frame_name as string,
      imageWidth: s.width as number,
      imageHeight: s.height as number,
    }),
}));
