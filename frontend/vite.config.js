import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = process.env.VITE_BACKEND_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/analyze': { target: BACKEND, changeOrigin: true },
      '/status':  { target: BACKEND, changeOrigin: true },
      '/results': { target: BACKEND, changeOrigin: true },
      '/reset':   { target: BACKEND, changeOrigin: true },
      '/health':  { target: BACKEND, changeOrigin: true },
    },
  },
})
