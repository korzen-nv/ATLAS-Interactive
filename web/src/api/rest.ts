import type { SessionInfo, WorkspaceInfo, PaletteEntry, AppConfig } from "./types";

const BASE_URL = "/api";

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${url}`, options);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return res.json();
}

export async function createSession(params: {
  workspace?: string;
  video?: string;
  images?: string;
}): Promise<SessionInfo> {
  const query = new URLSearchParams();
  if (params.workspace) query.set("workspace", params.workspace);
  if (params.video) query.set("video", params.video);
  if (params.images) query.set("images", params.images);
  return fetchJson(`/session?${query}`, { method: "POST" });
}

export async function uploadVideo(file: File): Promise<SessionInfo> {
  const form = new FormData();
  form.append("file", file);
  return fetchJson("/workspace/upload-video", { method: "POST", body: form });
}

export async function uploadImages(file: File): Promise<SessionInfo> {
  const form = new FormData();
  form.append("file", file);
  return fetchJson("/workspace/upload-images", { method: "POST", body: form });
}

export async function getWorkspaceInfo(): Promise<WorkspaceInfo> {
  return fetchJson("/workspace/info");
}

export async function getPalette(): Promise<PaletteEntry[]> {
  return fetchJson("/palette");
}

export async function getConfig(): Promise<AppConfig> {
  return fetchJson("/config");
}

export async function updateConfig(params: Record<string, number>): Promise<void> {
  const query = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    query.set(k, String(v));
  }
  await fetchJson(`/config?${query}`, { method: "PUT" });
}

export async function exportVideo(): Promise<{ status: string; path: string }> {
  return fetchJson("/export/video", { method: "POST" });
}

export async function exportBinaryMasks(): Promise<{ status: string; path: string }> {
  return fetchJson("/export/binary-masks", { method: "POST" });
}

export function getFrameUrl(ti: number, visMode?: string): string {
  let url = `${BASE_URL}/frame/${ti}`;
  if (visMode) url += `?vis_mode=${visMode}`;
  return url;
}
