<template>
  <div class="app-container">

    <navbar></navbar>
    <Header :title="currentTitle" :subtitle="currentSubtitle" />

    <main class="main-content">
      <router-view v-slot="{ Component, route }">
        <transition name="parallax" mode="out-in" @after-enter="scrollToTop">
          <suspense>
            <component :is="Component" :key="route.path" />
            <template #fallback>
              <div class="loading-spinner">
                <div class="spinner"></div>
              </div>
            </template>
          </suspense>
        </transition>
      </router-view>
    </main>

    <Footer />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import Header from '@/components/Header.vue'
import Footer from '@/components/Footer.vue'
import navbar from'@/components/NavBar.vue'

const route = useRoute()

const currentTitle = computed(() => route.meta.title || '默认标题')
const currentSubtitle = computed(() => route.meta.subtitle || '默认副标题')

const scrollToTop = () => {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  })
}
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #f8fafc;
}

.main-content {
  flex: 1;
  padding: 2rem 1rem;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
  position: relative;
}

.loading-spinner {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 4px solid rgba(0, 0, 0, 0.1);
  border-radius: 50%;
  border-top-color: #3498db;
  animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>

<style>
/* 视差效果动画 */
.parallax-enter-active {
  transition: all 0.6s cubic-bezier(0.22, 1, 0.36, 1);
  will-change: transform, opacity;
}
.parallax-leave-active {
  transition: all 0.4s cubic-bezier(0.22, 1, 0.36, 1);
  will-change: transform, opacity;
}
.parallax-enter-from {
  opacity: 0;
  transform: translateY(20px) scale(0.98);
}
.parallax-leave-to {
  opacity: 0;
  transform: translateY(-10px) scale(1.02);
}
</style>
