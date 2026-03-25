/**
 * Convert canvas pixel coordinates to image coordinates.
 * Replicates gui/gui.py pixel_pos_to_image_pos().
 */
export function pixelPosToImagePos(
  canvasX: number,
  canvasY: number,
  canvasWidth: number,
  canvasHeight: number,
  imageWidth: number,
  imageHeight: number
): { x: number; y: number } {
  const hRatio = canvasHeight / imageHeight;
  const wRatio = canvasWidth / imageWidth;
  const dominateRatio = Math.min(hRatio, wRatio);

  // Solve scale
  let x = canvasX / dominateRatio;
  let y = canvasY / dominateRatio;

  // Solve padding
  const fh = canvasHeight / dominateRatio;
  const fw = canvasWidth / dominateRatio;
  x -= (fw - imageWidth) / 2;
  y -= (fh - imageHeight) / 2;

  return { x, y };
}

export function clampToImage(
  x: number,
  y: number,
  imageWidth: number,
  imageHeight: number
): { x: number; y: number } {
  return {
    x: Math.max(0, Math.min(imageWidth - 1, x)),
    y: Math.max(0, Math.min(imageHeight - 1, y)),
  };
}

export function isOutOfBound(
  x: number,
  y: number,
  imageWidth: number,
  imageHeight: number
): boolean {
  return x < 0 || y < 0 || x > imageWidth - 1 || y > imageHeight - 1;
}
