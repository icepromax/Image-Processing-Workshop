import { createRouter, createWebHistory } from 'vue-router'
import resources from '@/components/ResourcesPage.vue'
import HomePage from "@/components/HomePage.vue";
const routes = [

    {
        path: '/resources',
        name: 'Resources',
        component: resources,
        meta: {
            title: 'GAN 学习资源中心',
            subtitle: '探索生成对抗网络的奥秘与应用'
        }
    },
    {
        path: '/home',
        name: 'HomePage',
        component: HomePage,
        meta: {
            title: 'AI 图像生成工坊',
            subtitle: '基于生成对抗网络(GAN)的智能图像生成'
        }
    },
    {
        path: '/',
        name: 'Home',
        component: HomePage,
        meta: {
            title: 'AI 图像生成工坊',
            subtitle: '基于生成对抗网络(GAN)的智能图像生成'
        }
    }
]

const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes,
    scrollBehavior(to, from, savedPosition) {
        if (savedPosition) {
            return savedPosition
        } else {
            return { top: 0 }
        }
    }
})

// 确保每次导航后都滚动到顶部
router.afterEach(() => {
    window.scrollTo({
        top: 0,
        behavior: 'instant' // 或 'smooth' 如果需要平滑滚动
    })
})

export default router
