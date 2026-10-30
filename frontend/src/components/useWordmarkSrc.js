import { useEffect, useState } from "react";

// ElectricLogo traces an image, and an SVG-as-image can't load web fonts,
// so rasterise the wordmark in the display font onto a canvas once the font is ready.
export default function useWordmarkSrc(text = "riffapart", font = "700 132px Qolians, Antonio, sans-serif") {
  const [src, setSrc] = useState(null);

  useEffect(() => {
    let alive = true;
    const draw = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.font = font;
      const m = ctx.measureText(text);
      const pad = 8;
      canvas.width = Math.ceil(m.width) + pad * 2;
      canvas.height = Math.ceil(m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) + pad * 2;
      ctx.font = font; // resizing the canvas resets its state
      ctx.fillStyle = "#fff";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(text, pad, pad + m.actualBoundingBoxAscent);
      if (alive) setSrc(canvas.toDataURL("image/png"));
    };
    (document.fonts?.load(font) ?? Promise.resolve()).then(draw, draw);
    return () => { alive = false; };
  }, [text, font]);

  return src;
}
