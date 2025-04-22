<!-- NavBar.vue -->
<template>
    <nav class="nav-container">
      <div class="nav-content">
        <router-link to="/home" class="logo">
          <span class="logo-text">GAN Lab</span>
        </router-link>

        <div class="nav-links">
          <router-link
              v-for="route in routes"
              :key="route.path"
              :to="route.path"
              class="nav-link"
              active-class="active"
          >
            {{ route.name }}
            <div class="nav-underline"></div>
          </router-link>
        </div>
      </div>
    </nav>
</template>

<script setup>
import { defineProps } from 'vue'

const props = defineProps({
  routes: {
    type: Array,
    default: () => [
      {
        path: '/home',
        name: '模型体验'
      },
      {
        path: '/resources',
        name: '学习资源'
      }
    ]
  }
})
</script>

<style scoped>

/* 新增导航栏样式 */
.nav-container {
  position: relative;
  z-index: 10;
  padding: 1rem 2rem;
  backdrop-filter: blur(5px);
  background-color: #0c0c10;
}

.nav-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
}

/* Logo样式 */
.logo {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  text-decoration: none;
}

.logo-icon {
  font-size: 2rem;
  margin-right: 0.75rem;
}

.logo-text {
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(90deg, #fff 0%, #8b5cf6 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

/* 导航链接样式 */
.nav-links {
  display: flex;
  gap: 2rem;
}

.nav-link {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: rgba(255, 255, 255, 0.8);
  text-decoration: none;
  font-weight: 500;
  padding: 0.8rem 1.2rem;
  transition: all 0.3s ease;
  border-radius: 4px;
}

.nav-icon {
  font-size: 1.2rem;
  transition: color 0.3s ease;
}

.nav-link:hover {
  color: white;
  background: rgba(99, 102, 241, 0.1);
}

.nav-link:hover .nav-icon {
  color: #8b5cf6;
}

.nav-link.active {
  color: white;
  font-weight: 600;
}

.nav-underline {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
  transform: scaleX(0);
  transform-origin: right;
  transition: transform 0.3s ease;
}

.nav-link:hover .nav-underline,
.nav-link.active .nav-underline {
  transform: scaleX(1);
  transform-origin: left;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .nav-container {
    padding: 1rem;
  }

  .nav-links {
    gap: 1rem;
  }

  .nav-link {
    padding: 0.5rem;
    font-size: 0.9rem;
  }
}
</style>
