import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { readFileSync } from 'node:fs'

const runtimeFile = new URL('../.cache/web-runtime.json', import.meta.url)

function backendTarget(): string {
  if (process.env.ROBOT_PORTRAIT_API_URL) return process.env.ROBOT_PORTRAIT_API_URL
  try {
    const state = JSON.parse(readFileSync(runtimeFile, 'utf8')) as { url?: string }
    if (state.url && /^http:\/\/(127\.0\.0\.1|localhost|\[::1\]|[\w.-]+):\d+$/.test(state.url)) {
      return state.url
    }
  } catch {
    // The API may not have started yet; retry on the next request.
  }
  return 'http://127.0.0.1:8000'
}

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: backendTarget(),
        changeOrigin: true,
        configure(proxy) {
          // Refresh before every request, including when the API starts after Vite.
          proxy.before('web', 'stream', (_req, _res, options) => {
            options.target = new URL(backendTarget())
          })
        },
      },
    },
  },
})
