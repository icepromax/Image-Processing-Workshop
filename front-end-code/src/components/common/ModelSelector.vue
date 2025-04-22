<template>
  <div class="model-selector">
    <label>选择修复模型:</label>
    <div class="model-buttons">
      <button
          v-for="model in models"
          :key="model.id"
          :class="{ active: selectedModel === model.id }"
          @click="selectModel(model)"
      >
        {{ model.name }}
        <span class="model-tag" :style="{ backgroundColor: model.tagColor }">
          {{ model.tag }}
        </span>
      </button>
    </div>
    <div class="model-info" v-if="selectedModelInfo">
      <div class="description-container">
        <p v-html="getModelDescription(selectedModel)"></p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, defineProps, defineEmits } from 'vue';

const props = defineProps({
  models: {
    type: Array,
    required: true
  }
});

const emit = defineEmits(['update:model']);

const selectedModel = ref(props.models[0]?.id);

const selectedModelInfo = computed(() => {
  return props.models.find(model => model.id === selectedModel.value);
});

const modelDescriptions = {
  'cyclegan': `本模型采用CycleGAN的模型架构训练，可实现照片到莫奈风格画作的基础转换，由于训练数据的构成，更加适用与风景照，并且输出为256*256的图片，可能会对输入图片进行裁剪处理。
    参考了<span class="external-link"><a href="https://blog.csdn.net/qq_39547794/article/details/125409710" target="_blank">CycleGAN的PyTorch实现</a></span>（代码含详细注释），共训练20个epoch。训练数据集由<span class="highlight">1193幅莫奈绘画</span>和<span class="highlight">7038张自然照片</span>组成，数据来源于<span class="external-link"><a href="https://tianchi.aliyun.com/dataset/93932" target="_blank">阿里云天池Monet2Photo数据集</a></span>。`,

  'real-esrgan': `本模型源于<span class="external-link"><a href="https://github.com/xinntao/Real-ESRGAN?tab=readme-ov-file#python-script">github项目Real-ESRGAN</a></span>,可实现通用图像的高清化。`,
  'Gan_unet': `本模型基于<span class="external-link"><a href="https://github.com/longfeibai/image-restoration">github项目image-restoration</a></span>修改进行训练后得到。
            基于gan和u-net的实现图像修复。所用数据集为<span class="external-link"><a href="https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data">CelebA人脸数据集</a></span>，可实现人脸缺失的图像修复，训练所得模型的泛化性能相对较差，提前标注好缺失部分
            修复效果会更好，在此直接对图像进行识别提取缺失部分进行修复，仅仅作为展示，还有很多基于gan的图像修复的技术，如<a href="https://github.com/tedqin/GAN-ImageRepairing">DCGAN</a>等等。`,
  'default': '该模型暂无详细描述信息。'
};


const getModelDescription = (modelId) => {
  return modelDescriptions[modelId] || modelDescriptions.default;
};

const navigateTo = (reference) => {
  // 这里可以扩展为实际导航逻辑
  console.log(`跳转到引用: ${reference}`);
};

const selectModel = (model) => {
  selectedModel.value = model.id;
  emit('update:model', model.id);
};
</script>

<style scoped>
.model-selector {
  margin-bottom: 2rem;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.model-selector label {
  display: block;
  margin-bottom: 0.75rem;
  font-weight: 600;
  color: #334155;
  font-size: 0.95rem;
}

.model-buttons {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.model-buttons button {
  padding: 0.75rem 1.25rem;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-weight: 500;
  color: #475569;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.2s ease;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.model-buttons button:hover {
  border-color: #c7d2fe;
  background: #f8fafc;
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.model-buttons button.active {
  background: #e0e7ff;
  border-color: #6366f1;
  color: #4f46e5;
  box-shadow: 0 1px 3px rgba(79, 70, 229, 0.2);
}

.model-tag {
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 9999px;
  color: white;
  font-weight: 600;
}

.model-info {
  margin-top: 1.25rem;
  padding: 1.25rem;
  background: #f8fafc;
  border-radius: 10px;
  font-size: 0.9rem;
  color: #475569;
  line-height: 1.6;
  border-left: 3px solid #6366f1;
  transition: all 0.3s ease;
}

.model-info:hover {
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.description-container {
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}

.reference {
  color: #6366f1;
  font-weight: 500;
  cursor: pointer;
  position: relative;
  padding: 0 2px;
  transition: all 0.2s ease;
}

.reference:hover {
  color: #4f46e5;
  background-color: #e0e7ff;
  border-radius: 3px;
}

.reference::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 100%;
  height: 1px;
  background-color: #c7d2fe;
  transform: scaleX(0);
  transition: transform 0.2s ease;
}

.reference:hover::after {
  transform: scaleX(1);
}
</style>
