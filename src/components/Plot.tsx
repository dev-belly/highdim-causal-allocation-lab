import { useEffect, useRef } from 'react';

/** 极简 Plotly 静态类型（仅取我们用到的 API，避免引入庞大的官方类型包） */
interface PlotlyType {
  newPlot: (
    el: HTMLElement,
    data: unknown[],
    layout?: Record<string, unknown>,
    config?: Record<string, unknown>,
  ) => Promise<unknown>;
  react: (
    el: HTMLElement,
    data: unknown[],
    layout?: Record<string, unknown>,
    config?: Record<string, unknown>,
  ) => Promise<unknown>;
  purge: (el: HTMLElement) => void;
}

declare global {
  interface Window {
    Plotly?: PlotlyType;
  }
}

interface PlotProps {
  data: unknown[];
  layout?: Record<string, unknown>;
  className?: string;
}

const CONFIG = { responsive: true, displayModeBar: false } as const;

/** 通用 Plotly 容器：深色主题、响应式、无工具栏。Plotly 由 CDN 注入 window.Plotly。 */
export function Plot({ data, layout, className }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  const plotted = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let cancelled = false;
    let tries = 0;

    const baseLayout: Record<string, unknown> = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#9fb0c9', family: 'Inter, -apple-system, sans-serif', size: 12 },
      margin: { t: 46, r: 18, b: 52, l: 64 },
      xaxis: { gridcolor: '#1f2a40', zerolinecolor: '#2c3852', linecolor: '#2c3852' },
      yaxis: { gridcolor: '#1f2a40', zerolinecolor: '#2c3852', linecolor: '#2c3852' },
      legend: { orientation: 'h', y: -0.18, font: { size: 11 } },
      hovermode: 'closest',
    };

    const attempt = () => {
      if (cancelled) return;
      const P = window.Plotly;
      if (!P) {
        // CDN 偶发慢加载：最多轮询 ~5s
        if (tries++ < 50) {
          window.setTimeout(attempt, 100);
        }
        return;
      }
      // 深合并：图表级 xaxis/yaxis/scene 不应覆盖基础深色网格与坐标轴配色
      const ov = (layout ?? {}) as Record<string, any>;
      const mergedLayout: Record<string, unknown> = {
        ...baseLayout,
        ...layout,
        xaxis: { ...(baseLayout.xaxis as Record<string, any>), ...(ov.xaxis || {}) },
        yaxis: { ...(baseLayout.yaxis as Record<string, any>), ...(ov.yaxis || {}) },
      };
      if (ov.scene) mergedLayout.scene = { bgcolor: 'rgba(0,0,0,0)', ...ov.scene };
      if (!plotted.current) {
        void P.newPlot(el, data, mergedLayout, CONFIG);
        plotted.current = true;
      } else {
        void P.react(el, data, mergedLayout, CONFIG);
      }
    };

    attempt();

    return () => {
      cancelled = true;
      if (plotted.current && window.Plotly) {
        window.Plotly.purge(el);
      }
      plotted.current = false;
    };
  }, [data, layout]);

  return <div ref={ref} className={className} style={{ width: '100%', height: '100%' }} />;
}

export const PALETTE = ['#4f9dff', '#22d3ee', '#34d399', '#fbbf24', '#f87171', '#a78bfa'];
