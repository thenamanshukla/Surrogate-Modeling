import { useEffect, useRef, useState } from "react";

export default function AnimatedNumber({ value, decimals = 4, suffix = "" }) {
  const [display, setDisplay] = useState(value);
  const previousValue = useRef(value);
  const frameRef = useRef(null);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (value === undefined || value === null) return;

    if (prefersReducedMotion) {
      setDisplay(value);
      previousValue.current = value;
      return;
    }

    const start = previousValue.current;
    const end = value;
    const duration = 700;
    const startTime = performance.now();

    function tick(now) {
      const elapsed = now - startTime;
      const t = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      const current = start + (end - start) * eased;
      setDisplay(current);
      if (t < 1) {
        frameRef.current = requestAnimationFrame(tick);
      } else {
        previousValue.current = end;
      }
    }

    frameRef.current = requestAnimationFrame(tick);
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [value]);

  if (value === undefined || value === null) return <span>n a</span>;

  return (
    <span>
      {display.toFixed(decimals)}
      {suffix}
    </span>
  );
}
