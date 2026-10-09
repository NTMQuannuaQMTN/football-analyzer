"use client";
import { useCallback, useEffect, useRef, useState } from "react";

/** Match clock driven by requestAnimationFrame. `stopAt` ends a replay window. */
export function useReplayClock(duration: number) {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(8);
  const [stopAt, setStopAt] = useState<number | null>(null);
  const tRef = useRef(0);
  const last = useRef<number | null>(null);

  useEffect(() => {
    if (!playing) {
      last.current = null;
      return;
    }
    let raf = 0;
    const tick = (now: number) => {
      if (last.current != null) {
        let nt = tRef.current + ((now - last.current) / 1000) * speed;
        let stop = false;
        if (stopAt != null && nt >= stopAt) {
          nt = stopAt;
          stop = true;
        }
        if (nt >= duration) {
          nt = duration;
          stop = true;
        }
        tRef.current = nt;
        setT(nt);
        if (stop) {
          setPlaying(false);
          setStopAt(null);
          return;
        }
      }
      last.current = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, stopAt, duration]);

  const seek = useCallback((nt: number) => {
    tRef.current = Math.max(0, Math.min(duration, nt));
    setT(tRef.current);
  }, [duration]);

  const playWindow = useCallback((start: number, end: number, spd = 1) => {
    seek(start);
    setSpeed(spd);
    setStopAt(end);
    setPlaying(true);
  }, [seek]);

  return { t, playing, setPlaying, speed, setSpeed, seek, playWindow, stopAt, setStopAt };
}
