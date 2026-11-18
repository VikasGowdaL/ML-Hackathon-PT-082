
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Listen on all interfaces so the UI can also be opened from another device on the LAN
  // (http://<this-laptop-ip>:5173). The API address is configured with VITE_API_URL.
  server: { host: true, port: 5173, strictPort: true },
  preview: { host: true, port: 5173, strictPort: true },
})
