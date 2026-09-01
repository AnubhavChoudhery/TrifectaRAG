import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5172,
    strictPort: true,
    proxy: {
      '/upload-pdf': 'http://127.0.0.1:8001',
      '/ingest-status': 'http://127.0.0.1:8001',
      '/ask': 'http://127.0.0.1:8001',
      '/image': 'http://127.0.0.1:8001',
      '/health': 'http://127.0.0.1:8001',
    },
  },
})
