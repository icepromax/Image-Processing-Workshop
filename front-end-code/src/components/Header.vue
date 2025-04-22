<template>

  <header class="app-header">
    <!-- 科技感背景元素 -->
    <div class="tech-lines">
      <div class="line"></div>
      <div class="line"></div>
      <div class="line"></div>
    </div>

    <!-- 动态粒子背景 -->
    <div class="particles" ref="particles"></div>


    <!-- 主要内容 -->
    <div class="header-content">
      <h1 class="title">
        <span class="title-text">{{ title }}</span>
        <span class="title-highlight"></span>
      </h1>
      <p class="subtitle">
        <span class="subtitle-text">{{ subtitle }}</span>
        <span class="typing-cursor">|</span>
      </p>
    </div>
  </header>
</template>


<script setup>
import { onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import {
  MagicStick,
  HomeFilled,
  CollectionTag
} from '@element-plus/icons-vue'

// 定义 props
const props = defineProps({
  title: {
    type: String,
    default: "AI 图像生成工坊" // 默认值
  },
  subtitle: {
    type: String,
    default: "基于生成对抗网络(GAN)的智能图像生成" // 默认值
  }
})



const route = useRoute()

// 路由配置
const routes = [
  {
    path: '/home',
    name: '工作台',
    icon: HomeFilled
  },
  {
    path: '/resources',
    name: '资源中心',
    icon: CollectionTag
  }
]

const particles = ref(null)

onMounted(() => {
  // 简单的粒子动画效果
  if (particles.value) {
    const particleCount = 30
    for (let i = 0; i < particleCount; i++) {
      const particle = document.createElement('div')
      particle.className = 'particle'
      particle.style.left = `${Math.random() * 100}%`
      particle.style.top = `${Math.random() * 100}%`
      particle.style.width = `${Math.random() * 3 + 1}px`
      particle.style.height = particle.style.width
      particle.style.animationDelay = `${Math.random() * 5}s`
      particles.value.appendChild(particle)
    }
  }
})
</script>



<style scoped>
.app-header {
  position: relative;
  text-align: center;
  padding: 4rem;
  background: linear-gradient(135deg, #0f0c29 0%, #0c0c10 50%, #24243e 100%);
  color: white;
  overflow: hidden;
  border-bottom: 1px solid rgba(99, 102, 241, 0.3);
}
.app-header1 {
  position: relative;
  text-align: center;
  background: linear-gradient(135deg, #0f0c29 0%, #0c0c10 50%, #24243e 100%);
  color: white;
  overflow: hidden;
  border-bottom: 1px solid rgba(138, 138, 159, 0.3);
}


/* 科技线条背景 */
.tech-lines {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0.15;
}

.tech-lines .line {
  position: absolute;
  height: 100%;
  width: 1px;
  background: linear-gradient(to bottom, transparent, #6366f1, transparent);
}

.tech-lines .line:nth-child(1) {
  left: 20%;
}

.tech-lines .line:nth-child(2) {
  left: 50%;
}

.tech-lines .line:nth-child(3) {
  left: 80%;
}

/* 粒子动画 */
.particles {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.particle {
  position: absolute;
  background-color: rgba(99, 102, 241, 0.6);
  border-radius: 50%;
  animation: float 15s infinite linear;
}

@keyframes float {
  0% {
    transform: translateY(0) translateX(0);
    opacity: 0;
  }
  10% {
    opacity: 1;
  }
  90% {
    opacity: 1;
  }
  100% {
    transform: translateY(-100vh) translateX(20px);
    opacity: 0;
  }
}

/* 标题样式 */
.header-content {
  position: relative;
  z-index: 2;
  max-width: 800px;
  margin: 0 auto;
}

.title {
  font-size: 3rem;
  font-weight: 800;
  margin-bottom: 1rem;
  background: linear-gradient(90deg, #fff 0%, #8b5cf6 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  position: relative;
  display: inline-block;
}

.title-highlight {
  position: absolute;
  bottom: -5px;
  left: 0;
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
  transform: scaleX(0);
  transform-origin: left;
  animation: highlight 2s ease-in-out infinite alternate;
}

@keyframes highlight {
  to {
    transform: scaleX(1);
  }
}

/* 副标题打字机效果 */
.subtitle {
  font-size: 1.3rem;
  color: #a5b4fc;
  max-width: 600px;
  margin: 0 auto 2rem;
  position: relative;
}

.typing-cursor {
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0;
  }
}

/* 科技感圆点装饰 */
.tech-dots {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 2rem;
}

.tech-dots .dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: #6366f1;
  opacity: 0;
  animation: pulse 2s infinite ease-in-out;
}

.tech-dots .dot:nth-child(1) {
  animation-delay: 0s;
}

.tech-dots .dot:nth-child(2) {
  animation-delay: 0.3s;
}

.tech-dots .dot:nth-child(3) {
  animation-delay: 0.6s;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(0.8);
    opacity: 0.5;
  }
  50% {
    transform: scale(1.2);
    opacity: 1;
  }
}

/* 滚动指示器 */
.scroll-indicator {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
}

.chevron {
  width: 20px;
  height: 20px;
  border-bottom: 2px solid #a5b4fc;
  border-right: 2px solid #a5b4fc;
  transform: rotate(45deg);
  margin: -10px auto;
  animation: scroll 2s infinite;
  opacity: 0;
}

.chevron:nth-child(1) {
  animation-delay: 0s;
}

.chevron:nth-child(2) {
  animation-delay: 0.2s;
}

.chevron:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes scroll {
  0% {
    opacity: 0;
    transform: rotate(45deg) translate(0, 0);
  }
  50% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    transform: rotate(45deg) translate(0, 10px);
  }
}

/* 响应式设计 */
@media (max-width: 768px) {
  .title {
    font-size: 2rem;
  }

  .subtitle {
    font-size: 1rem;
  }
}
</style>
