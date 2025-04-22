import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  server:{
    host:"0.0.0.0",
    port:8081,
    strictPort:true,
    https:false,
    allowedHosts: [
      'qrc9g4.natappfree.cc',  // 添加你的natapp域名
      'localhost'              // 保留本地访问
    ],
  },
})
