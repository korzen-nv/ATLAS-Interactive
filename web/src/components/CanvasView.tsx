import { useRef, useEffect, useCallback } from "react";
import { useAppStore } from "../state/store";
import { pixelPosToImagePos, clampToImage, isOutOfBound } from "../utils/coordinates";
import type { ClientMessage } from "../api/types";

interface Props {
  send: (msg: ClientMessage) => void;
}

export function CanvasView({ send }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const frameImage = useAppStore((s) => s.frameImage);
  const imageWidth = useAppStore((s) => s.imageWidth);
  const imageHeight = useAppStore((s) => s.imageHeight);
  const polygonMode = useAppStore((s) => s.polygonMode);
  const polygonPoints = useAppStore((s) => s.polygonPoints);
  const hoverFirst = useAppStore((s) => s.hoverFirst);
  const currentObject = useAppStore((s) => s.currentObject);
  const palette = useAppStore((s) => s.palette);

  // Draw the frame image on canvas
  useEffect(() => {
    if (!frameImage || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      // Size canvas to container
      const container = containerRef.current;
      if (container) {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
      }
      // Draw with aspect ratio preservation
      const { dx, dy, dw, dh } = fitImage(img.width, img.height, canvas.width, canvas.height);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, dx, dy, dw, dh);
    };
    img.src = `data:image/jpeg;base64,${frameImage}`;
  }, [frameImage]);

  // Draw polygon overlay
  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;

    const canvas = canvasRef.current;
    if (canvas) {
      overlay.width = canvas.width;
      overlay.height = canvas.height;
    }

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!polygonMode || polygonPoints.length === 0) return;

    const objColor = palette.find((p) => p.id === currentObject)?.color ?? [255, 0, 0];
    const [r, g, b] = objColor;

    // Convert image coords to canvas coords for drawing
    const canvasPoints = polygonPoints.map(([px, py]) =>
      imageToCanvas(px, py, imageWidth, imageHeight, overlay.width, overlay.height)
    );

    // Draw lines
    if (canvasPoints.length > 1) {
      ctx.strokeStyle = `rgb(${r},${g},${b})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(canvasPoints[0].x, canvasPoints[0].y);
      for (let i = 1; i < canvasPoints.length; i++) {
        ctx.lineTo(canvasPoints[i].x, canvasPoints[i].y);
      }
      ctx.stroke();
    }

    // Draw points
    canvasPoints.forEach((pt, i) => {
      ctx.beginPath();
      if (i === 0 && hoverFirst) {
        ctx.fillStyle = "white";
        ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
      } else {
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      }
      ctx.fill();
    });
  }, [polygonMode, polygonPoints, hoverFirst, currentObject, palette, imageWidth, imageHeight]);

  const getImageCoords = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || !imageWidth || !imageHeight) return null;
      const rect = canvas.getBoundingClientRect();
      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;
      const pos = pixelPosToImagePos(canvasX, canvasY, canvas.width, canvas.height, imageWidth, imageHeight);
      if (isOutOfBound(pos.x, pos.y, imageWidth, imageHeight)) return null;
      return clampToImage(pos.x, pos.y, imageWidth, imageHeight);
    },
    [imageWidth, imageHeight]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      const pos = getImageCoords(e);
      if (!pos) return;

      let action: "left" | "right" | "middle";
      if (e.button === 0) action = "left";
      else if (e.button === 2) action = "right";
      else if (e.button === 1) action = "middle";
      else return;

      send({ type: "click", action, x: Math.round(pos.x), y: Math.round(pos.y) });
    },
    [send, getImageCoords]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const pos = getImageCoords(e);
      if (!pos) return;
      send({ type: "mouse_move", x: Math.round(pos.x), y: Math.round(pos.y) });
    },
    [send, getImageCoords]
  );

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
  }, []);

  // Resize handler
  useEffect(() => {
    const handleResize = () => {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      if (!container || !canvas || !imageRef.current) return;
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const img = imageRef.current;
      const { dx, dy, dw, dh } = fitImage(img.width, img.height, canvas.width, canvas.height);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, dx, dy, dw, dh);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div ref={containerRef} className="canvas-container">
      <canvas
        ref={canvasRef}
        className="main-canvas"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onContextMenu={handleContextMenu}
      />
      <canvas ref={overlayRef} className="overlay-canvas" />
    </div>
  );
}

function fitImage(
  imgW: number,
  imgH: number,
  canvasW: number,
  canvasH: number
): { dx: number; dy: number; dw: number; dh: number } {
  const scale = Math.min(canvasW / imgW, canvasH / imgH);
  const dw = imgW * scale;
  const dh = imgH * scale;
  const dx = (canvasW - dw) / 2;
  const dy = (canvasH - dh) / 2;
  return { dx, dy, dw, dh };
}

function imageToCanvas(
  imgX: number,
  imgY: number,
  imgW: number,
  imgH: number,
  canvasW: number,
  canvasH: number
): { x: number; y: number } {
  const scale = Math.min(canvasW / imgW, canvasH / imgH);
  const dx = (canvasW - imgW * scale) / 2;
  const dy = (canvasH - imgH * scale) / 2;
  return {
    x: imgX * scale + dx,
    y: imgY * scale + dy,
  };
}
