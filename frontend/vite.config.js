import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5172,
    strictPort: true,
    proxy: {
      '/upload-pdf': { target: 'http://127.0.0.1:8001', timeout: 360_000, proxyTimeout: 360_000 },
      '/upload': { target: 'http://127.0.0.1:8001', timeout: 360_000, proxyTimeout: 360_000 },
      '/attach': { target: 'http://127.0.0.1:8001', timeout: 360_000, proxyTimeout: 360_000 },
      '/corpora': { target: 'http://127.0.0.1:8001', timeout: 360_000, proxyTimeout: 360_000 },
      '/settings': { target: 'http://127.0.0.1:8001' },
      '/ingest-status': { target: 'http://127.0.0.1:8001' },
      '/ask': { target: 'http://127.0.0.1:8001', timeout: 360_000, proxyTimeout: 360_000 },
      '/chat': { target: 'http://127.0.0.1:8001', timeout: 360_000, proxyTimeout: 360_000 },
      '/image': { target: 'http://127.0.0.1:8001' },
      '/health': { target: 'http://127.0.0.1:8001' },
    },
  },
})
