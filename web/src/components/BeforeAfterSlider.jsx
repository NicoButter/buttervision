import { useState } from "react";

export default function BeforeAfterSlider({ original, result }) {
  const [position, setPosition] = useState(50);
  if (!original) {
    return <div className="comparison-empty">El resultado aparecerá aquí cuando la edición esté integrada.</div>;
  }
  if (result) {
    return (
      <div className="before-after">
        <div className="comparison-canvas">
          <img src={original} alt="Imagen original" />
          <div className="comparison-overlay" style={{ clipPath: `inset(0 0 0 ${position}%)` }}>
            <img src={result} alt="Imagen editada" />
          </div>
          <span className="comparison-label original-label">Original</span>
          <span className="comparison-label result-label">Resultado</span>
        </div>
        <label className="comparison-control">
          <span className="visually-hidden">Comparar imagen original y resultado</span>
          <input
            type="range"
            min="0"
            max="100"
            value={position}
            aria-label="Comparar imagen original y resultado"
            onChange={(event) => setPosition(Number(event.target.value))}
          />
        </label>
      </div>
    );
  }
  return (
    <div className="comparison-grid">
      <figure><img src={original} alt="Imagen original" /><figcaption>Original</figcaption></figure>
      <figure><div className="comparison-empty">Resultado pendiente</div><figcaption>Resultado</figcaption></figure>
    </div>
  );
}
