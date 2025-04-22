<template>
  <footer class="site-footer">
    <!-- 波浪形分隔线 -->
    <div class="footer-wave">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" preserveAspectRatio="none">
        <path fill="currentColor" fill-opacity="1" d="M0,256L48,261.3C96,267,192,277,288,250.7C384,224,480,160,576,160C672,160,768,224,864,218.7C960,213,1056,139,1152,117.3C1248,96,1344,128,1392,144L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
      </svg>
    </div>

    <div class="footer-container">
      <!-- 主要内容 -->
      <div class="footer-content">
        <!-- 左侧: 关于我们 -->
        <div class="footer-about">
          <div class="footer-logo">
<!--            <span class="logo-icon">🖼️</span>-->
            <span class="logo-text">基于GAN的AI图像生成工坊</span>
          </div>
          <p class="footer-description">
            基于前沿生成对抗网络技术，为您的照片提供智能修复与增强服务。
          </p>
          <div class="footer-social">
            <a href="https://github.com/icepromax/Image-Processing-Workshop" class="social-link" aria-label="GitHub">
              <svg class="social-icon" viewBox="0 0 24 24">
                <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
              </svg>
            </a>
          </div>
        </div>

        <!-- 中间: 导航链接 -->
        <div class="footer-nav">
          <div class="nav-column">
            <h3 class="nav-title">探索</h3>
            <ul class="nav-links">
              <li><router-link to="/">模型体验</router-link></li>
              <li><router-link to="/resources">学习资料</router-link></li>
            </ul>
          </div>

        </div>

        <!-- 右侧: 了解更多 -->
        <div class="footer-contact">
          <h3 class="contact-title">了解更多</h3>
          <img src="@/assets/images/草料二维码.png" alt="CycleGAN架构" class="caoliao-img">
          <p class="footer-description">扫描二维码了解更多详细内容(内附代码和课后习题)</p>
        </div>
      </div>

      <!-- 底部版权 -->
      <div class="footer-bottom">
<!--        <div class="footer-legal">-->
<!--          <router-link to="/privacy">隐私政策</router-link>-->
<!--          <span class="divider">•</span>-->
<!--          <router-link to="/terms">服务条款</router-link>-->
<!--          <span class="divider">•</span>-->
<!--          <router-link to="/cookies">Cookie政策</router-link>-->
<!--        </div>-->
        <p class="copyright">
          &copy; {{ currentYear }} AI图像生成工坊.
        </p>
        <div class="footer-badge">
          <span class="badge">Deep Learning</span>
          <span class="badge">GAN</span>
        </div>
      </div>
    </div>
  </footer>
</template>

<script setup>
import { computed } from 'vue';

const currentYear = computed(() => new Date().getFullYear());
</script>

<style scoped>
.caoliao-img {
  width: 120px; /* 或你需要的大小，例如 100px、150px */
  height: auto; /* 保持图片宽高比例 */
  margin-top: 8px;
}

/* 基础样式 */
.site-footer {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  color: #e6e6e6;
  position: relative;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* 波浪形分隔线 */
.footer-wave {
  position: absolute;
  top: -50px;
  left: 0;
  width: 100%;
  height: 50px;
  color: #1a1a2e;
}

.footer-wave svg {
  width: 100%;
  height: 100%;
}

/* 容器布局 */
.footer-container {
  max-width: 1280px;
  margin: 0 auto;
  padding: 4rem 2rem 2rem;
}

.footer-content {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 3rem;
  margin-bottom: 3rem;
}

/* 关于我们部分 */
.footer-about {
  display: flex;
  flex-direction: column;
}

.footer-logo {
  display: flex;
  align-items: center;
  margin-bottom: 1.5rem;
}



.logo-text {
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(90deg, #3498db, #8e44ad);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.footer-description {
  color: #b3b3b3;
  line-height: 1.6;
  margin-bottom: 2rem;
}

.footer-social {
  display: flex;
  gap: 1rem;
}

.social-link {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  transition: all 0.3s ease;
}

.social-link:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-3px);
}

.social-icon {
  width: 20px;
  height: 20px;
  fill: currentColor;
}

/* 导航链接部分 */
.footer-nav {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 2rem;
}

.nav-column {
  display: flex;
  flex-direction: column;
}

.nav-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  color: #ffffff;
  position: relative;
  padding-bottom: 0.5rem;
}

.nav-title::after {
  content: '';
  position: absolute;
  left: 0;
  bottom: 0;
  width: 40px;
  height: 2px;
  background: linear-gradient(90deg, #3498db, transparent);
}

.nav-links {
  list-style: none;
  padding: 0;
  margin: 0;
}

.nav-links li {
  margin-bottom: 0.8rem;
}

.nav-links a {
  color: #b3b3b3;
  text-decoration: none;
  transition: all 0.3s ease;
  position: relative;
  padding-left: 0;
}

.nav-links a:hover {
  color: #3498db;
  padding-left: 10px;
}

.nav-links a::before {
  content: '→';
  position: absolute;
  left: -10px;
  opacity: 0;
  transition: all 0.3s ease;
}

.nav-links a:hover::before {
  left: 0;
  opacity: 1;
}

/* 联系方式部分 */
.footer-contact {
  display: flex;
  flex-direction: column;
}

.contact-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  color: #ffffff;
  position: relative;
  padding-bottom: 0.5rem;
}

.contact-title::after {
  content: '';
  position: absolute;
  left: 0;
  bottom: 0;
  width: 40px;
  height: 2px;
  background: linear-gradient(90deg, #8e44ad, transparent);
}

.contact-method {
  display: flex;
  align-items: center;
  margin-bottom: 1rem;
  color: #b3b3b3;
}

.contact-icon {
  width: 20px;
  height: 20px;
  margin-right: 1rem;
  fill: currentColor;
}

/* 底部版权部分 */
.footer-bottom {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  padding-top: 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.footer-legal {
  display: flex;
  gap: 1.5rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
  justify-content: center;
}

.footer-legal a {
  color: #b3b3b3;
  text-decoration: none;
  transition: color 0.3s ease;
  font-size: 0.9rem;
}

.footer-legal a:hover {
  color: #3498db;
}

.divider {
  color: rgba(255, 255, 255, 0.2);
}

.copyright {
  color: #7f8c8d;
  font-size: 0.9rem;
  margin-bottom: 1rem;
  text-align: center;
}

.footer-badge {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.badge {
  background: rgba(52, 152, 219, 0.2);
  color: #3498db;
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 500;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .footer-content {
    grid-template-columns: 1fr;
    gap: 2rem;
  }

  .footer-nav {
    grid-template-columns: 1fr;
  }

  .footer-legal {
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
  }

  .divider {
    display: none;
  }
}
</style>
