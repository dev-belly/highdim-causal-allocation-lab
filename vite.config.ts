import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// 专业前端工程配置：React + TS + 并行 Web Worker + Vitest
export default defineConfig({
  plugins: [react()],
  // 相对 base 使产物可部署在 GitHub Pages 任意子路径（如 /repo/）
  base: './',
  worker: {
    format: 'es',
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    target: 'es2020',
    chunkSizeWarningLimit: 1500,
  },
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
