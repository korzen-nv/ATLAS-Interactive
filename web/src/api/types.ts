export interface PaletteEntry {
  id: number;
  name: string;
  color: [number, number, number];
}

export interface SessionInfo {
  session_id: string;
  workspace: string;
  total_frames: number;
  width: number;
  height: number;
  num_objects?: number;
}

export interface WorkspaceInfo {
  workspace: string;
  total_frames: number;
  width: number;
  height: number;
  num_objects: number;
  frame_names: string[];
}

export interface AppConfig {
  mem_every: number;
  work_mem_min: number;
  work_mem_max: number;
  long_mem_max: number;
  output_fps: number;
  output_bitrate: number;
  vis_mode: string;
}

export interface MemoryStatus {
  perm_tokens: number;
  work_tokens: number;
  max_work_tokens: number;
  long_tokens: number;
  max_long_tokens: number;
  gpu_used_gb: number;
  gpu_total_gb: number;
  torch_used_gb: number;
}

// WebSocket messages from server
export type ServerMessage =
  | { type: "frame"; ti: number; image: string; name: string; total_frames: number }
  | { type: "state"; current_frame: number; total_frames: number; current_object: number;
      vis_mode: string; propagating: boolean; propagate_direction: string;
      polygon_mode: boolean; polygon_points: [number, number][];
      playing: boolean; frame_name: string; width: number; height: number }
  | { type: "console"; text: string }
  | { type: "progress"; value: number }
  | { type: "propagation_state"; propagating: boolean; direction: string }
  | { type: "memory_status" } & MemoryStatus
  | { type: "polygon_update"; points: [number, number][]; hover_first: boolean; polygon_mode: boolean };

// WebSocket messages to server
export type ClientMessage =
  | { type: "click"; action: "left" | "right" | "middle"; x: number; y: number }
  | { type: "mouse_move"; x: number; y: number }
  | { type: "navigate"; frame: number }
  | { type: "propagate"; direction: "forward" | "backward" }
  | { type: "pause" }
  | { type: "commit" }
  | { type: "set_object"; id: number }
  | { type: "set_vis_mode"; mode: string }
  | { type: "toggle_vis_mode" }
  | { type: "reset_frame" }
  | { type: "reset_object" }
  | { type: "clear_memory"; permanent: boolean }
  | { type: "play_video"; playing: boolean }
  | { type: "update_config"; work_mem_min?: number; work_mem_max?: number;
      long_mem_max?: number; mem_every?: number }
  | { type: "get_memory_status" }
  | { type: "get_state" };
