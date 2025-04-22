import { createApp } from 'vue'
import App from './App.vue'
import axios from 'axios'
import router from './router'

// 创建Vue实例
const app = createApp(App)

// 全局配置axios
axios.defaults.baseURL = 'http://localhost:5000'
app.config.globalProperties.$http = axios
app.use(router)


app.mount('#app')
