<template>
  <section class="intro-section">
    <div class="intro-content">
      <h2>让残缺的记忆重获新生</h2>
      <p>我们的AI技术能够智能处理各种图像问题，通过先进的深度学习算法实现专业级图像处理效果。</p>

      <div class="tech-grid">
        <div class="tech-card" v-for="tech in techList" :key="tech.title">
          <div class="tech-icon">{{ tech.icon }}</div>
          <h3>{{ tech.title }}</h3>
          <p>{{ tech.desc }}</p>
        </div>
      </div>

      <!-- 模型对比系统 -->
      <div class="model-system">
        <div class="model-selector">
          <button
              v-for="model in models"
              :key="model.id"
              :class="{ active: activeModel === model.id }"
              @click="switchModel(model.id)"
          >
            <span class="model-icon">{{ model.icon }}</span>
            {{ model.name }}
          </button>
        </div>

        <div class="comparison-system">
          <div class="comparison-card" v-for="(example, exIndex) in getActiveModel.examples" :key="exIndex">
            <h4>{{ example.title }}</h4>
            <div class="image-comparison">
              <div class="image-box">
                <div class="image-label">原始图片</div>
                <img :src="example.original" :alt="`${getActiveModel.name}原始示例${exIndex + 1}`">
              </div>
              <div class="arrow-animation">⇒</div>
              <div class="image-box">
                <div class="image-label">{{ getActiveModel.name }}效果</div>
                <img :src="example.processed" :alt="`${getActiveModel.name}处理示例${exIndex + 1}`">
              </div>
            </div>
            <p class="example-desc">{{ example.description }}</p>
          </div>
        </div>

        <div class="model-features">
          <h3>{{ getActiveModel.name }}技术特点</h3>
          <ul>
            <li v-for="(feature, fIndex) in getActiveModel.features" :key="fIndex">
              <span class="feature-icon">✓</span> {{ feature }}
            </li>
          </ul>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, computed } from 'vue';

const activeModel = ref('superres');

// 技术特点展示
const techList = [
  { icon: '🛠️', title: '多模型架构', desc: '针对不同场景专项优化' },
  { icon: '⚡', title: '高效处理', desc: '快速获得专业级效果' },
  { icon: '🎯', title: '精准识别', desc: '智能分析图像内容' }
];

// 模型数据
const models = [
  {
    id: 'superres',
    name: '图像高清化',
    icon: '🔍',
    description: '提升图像分辨率并增强细节',
    features: [
      '4K/8K超分辨率重建',
      '细节增强算法',
      '噪点智能抑制',
      '边缘锐化技术'
    ],
    examples: [
      {
        title: '低分辨率增强',
        original: new URL('@/assets/images/图像高清化input.png', import.meta.url).href,
        processed: new URL('@/assets/images/图像高清化output.jpg', import.meta.url).href,
        description: '将低分辨率图像提升至4K画质并增强细节'
      }
    ]
  },
  {
    id: 'restoration',
    name: '图像残缺修复',
    icon: '🧩',
    description: '智能修复破损、划痕、缺失区域的图像',
    features: [
      '破损区域智能识别',
      '自然纹理生成技术',
      '多尺度细节修复',
      '色彩一致性保持'
    ],
    examples: [
      {
        title: '破损图像修复',
        original: new URL('@/assets/images/图像修复input.png', import.meta.url).href,
        processed: new URL('@/assets/images/图像修复output.png', import.meta.url).href,
        description: '自动修复图像缺失'
      }
    ]
  },

  {
    id: 'style',
    name: '图像风格迁移',
    icon: '🎨',
    description: '将图像转换为各种艺术风格',
    features: [
      '多种艺术风格选择',
      '内容-风格分离算法',
      '笔触细节保留',
      '实时风格预览'
    ],
    examples: [
      {
        title: '油画风格',
        original: new URL('@/assets/images/cyclegan原图.png', import.meta.url).href,
        processed: new URL('@/assets/images/cyclegan转化图.png', import.meta.url).href,
        description: '将普通照片转换为梵高风格的油画效果'
      }
    ]
  }
];

const getActiveModel = computed(() => {
  return models.find(model => model.id === activeModel.value);
});

const switchModel = (modelId) => {
  activeModel.value = modelId;
};
</script>

<style scoped>
.intro-section {
  max-width: 1200px;
  margin: 0 auto;
  padding: 60px 20px;
}

.intro-content {
  width: 100%;
}

h2 {
  font-size: 2.2rem;
  color: #333;
  margin-bottom: 20px;
}

p {
  color: #555;
  line-height: 1.6;
  margin-bottom: 30px;
}

.tech-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 25px;
  margin: 40px 0;
}

.tech-card {
  background: white;
  border-radius: 12px;
  padding: 25px;
  text-align: center;
  box-shadow: 0 5px 20px rgba(0,0,0,0.08);
  transition: transform 0.3s, box-shadow 0.3s;
}

.tech-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

.tech-icon {
  font-size: 2.5rem;
  margin-bottom: 15px;
}

.tech-card h3 {
  font-size: 1.2rem;
  margin-bottom: 10px;
  color: #333;
}

/* 模型系统样式 */
.model-system {
  margin-top: 50px;
  background: white;
  border-radius: 16px;
  padding: 30px;
  box-shadow: 0 5px 25px rgba(0,0,0,0.1);
}

.model-selector {
  display: flex;
  gap: 15px;
  margin-bottom: 30px;
  flex-wrap: wrap;
}

.model-selector button {
  padding: 12px 25px;
  border: none;
  border-radius: 30px;
  background: #f5f7fa;
  cursor: pointer;
  transition: all 0.3s;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1rem;
}

.model-selector button.active {
  background: #4a6bff;
  color: white;
}

.model-selector button:hover {
  transform: translateY(-2px);
}

.model-icon {
  font-size: 1.2rem;
}

/* 对比系统 */
.comparison-system {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 30px;
  margin-bottom: 40px;
}

.comparison-card {
  background: #f9fafc;
  border-radius: 12px;
  padding: 20px;
  transition: transform 0.3s;
}

.comparison-card:hover {
  transform: translateY(-5px);
}

.comparison-card h4 {
  font-size: 1.1rem;
  color: #444;
  margin-bottom: 15px;
  text-align: center;
}

.image-comparison {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
  margin-bottom: 15px;
}

.image-box {
  position: relative;
  width: 100%;
  max-width: 250px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 3px 15px rgba(0,0,0,0.1);
}

.image-box img {
  width: 100%;
  height: auto;
  display: block;
}

.image-label {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0,0,0,0.7);
  color: white;
  padding: 8px;
  text-align: center;
  font-size: 0.9rem;
}

.arrow-animation {
  font-size: 1.8rem;
  color: #4a6bff;
  animation: pulse 1.5s infinite;
}

.example-desc {
  color: #666;
  font-size: 0.95rem;
  text-align: center;
  margin-top: 10px;
}

/* 技术特点 */
.model-features {
  background: #f5f7fa;
  border-radius: 12px;
  padding: 25px;
}

.model-features h3 {
  text-align: center;
  margin-bottom: 20px;
  color: #333;
  font-size: 1.3rem;
}

.model-features ul {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 15px;
  padding: 0;
  list-style: none;
}

.model-features li {
  background: white;
  padding: 12px 15px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  gap: 10px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.feature-icon {
  color: #4a6bff;
  font-weight: bold;
}

@keyframes pulse {
  0% { opacity: 0.7; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.1); }
  100% { opacity: 0.7; transform: scale(1); }
}

@media (max-width: 768px) {
  .tech-grid {
    grid-template-columns: 1fr;
  }

  .comparison-system {
    grid-template-columns: 1fr;
  }

  .image-comparison {
    flex-direction: column;
  }

  .arrow-animation {
    transform: rotate(90deg);
    margin: 10px 0;
  }
}

/* 在原有样式基础上修改或添加以下内容 */

.image-comparison {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
  margin-bottom: 15px;
  flex-wrap: wrap; /* 添加换行以适应小屏幕 */
}

.image-box {
  position: relative;
  width: 100%;
  max-width: 300px; /* 增大最大宽度 */
  height: 300px; /* 固定高度 */
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 3px 15px rgba(0,0,0,0.1);
}

.image-box img {
  width: 100%;
  height: 100%;
  object-fit: cover; /* 保持图片比例并填充容器 */
  display: block;
  transition: transform 0.3s ease; /* 添加悬停效果 */
}

.image-box:hover img {
  transform: scale(1.03); /* 悬停时轻微放大 */
}

/* 调整移动端布局 */
@media (max-width: 768px) {
  .image-box {
    max-width: 100%;
    height: 250px; /* 移动端稍小的高度 */
  }

  .arrow-animation {
    transform: rotate(90deg);
    margin: 15px 0;
    font-size: 2rem; /* 增大箭头大小 */
  }

  .comparison-card {
    padding: 15px;
  }
}

/* 添加图片加载过渡效果 */
.image-box img {
  background: linear-gradient(110deg, #f5f5f5 8%, #eee 18%, #f5f5f5 33%);
  background-size: 200% 100%;
  animation: 1.5s shine linear infinite;
}

@keyframes shine {
  to {
    background-position-x: -200%;
  }
}
</style>
