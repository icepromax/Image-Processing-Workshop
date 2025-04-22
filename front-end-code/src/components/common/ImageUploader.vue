<template>
  <div class="upload-container">
    <div
        class="upload-box"
        @click="triggerFileInput"
        @dragover.prevent
        @drop="handleDrop"
        :class="{ 'has-image': imagePreview }"
    >
      <input
          type="file"
          ref="fileInput"
          @change="handleFileChange"
          accept="image/*"
          hidden
      >
      <template v-if="!imagePreview">
        <div class="upload-icon">
          <svg viewBox="0 0 24 24" width="48" height="48">
            <path d="M19,13H13V19H11V13H5V11H11V5H13V11H19V13Z" fill="currentColor"/>
          </svg>
        </div>
        <p>点击或拖拽上传图片</p>
        <p class="upload-hint">支持 JPG, PNG 格式，最大 10MB</p>
      </template>
      <div v-else class="image-preview">
        <img :src="imagePreview" alt="预览图">
        <button @click.stop="clearImage" class="clear-btn">
          <svg viewBox="0 0 24 24" width="20" height="20">
            <path d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" fill="currentColor"/>
          </svg>
        </button>
      </div>
    </div>

    <div class="sample-images" v-if="samples && samples.length">
      <p>或尝试示例图片:</p>
      <div class="sample-grid">
        <div
            v-for="sample in samples"
            :key="sample.id"
            class="sample-item"
            @click="handleSampleClick(sample)"

        >
          <img :src="sample.thumb" :alt="sample.name">
          <div class="sample-overlay">{{ sample.name }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, defineProps, defineEmits } from 'vue';

const props = defineProps({
  samples: {
    type: Array,
    default: () => []
  }
});

const emit = defineEmits(['file-upload', 'select-sample', 'clear']);

const fileInput = ref(null);
const imagePreview = ref(null);

const triggerFileInput = () => {
  fileInput.value.click();
};

const handleFileChange = (e) => {
  const file = e.target.files[0];
  if (file && file.type.match('image.*')) {
    processImageFile(file);
  }
};


const loadingSample = ref(null);
const handleSampleClick = async (sample) => {
  try {
    loadingSample.value = sample.id;

    // 1. 获取示例图片文件
    const file = await fetchSampleFile(sample);

    // 2. 处理文件（与普通上传相同）
    processImageFile(file);

  } catch (error) {
    console.error('示例加载失败:', error);
    alert('示例图片加载失败，请稍后重试');
  } finally {
    loadingSample.value = null;
  }
};

const fetchSampleFile = async (sample) => {
  // 从示例数据获取高清图URL
  const response = await fetch(sample.full);
  if (!response.ok) throw new Error('图片加载失败');

  // 将响应转换为File对象
  const blob = await response.blob();
  return new File([blob], `sample-${sample.id}.jpg`, {
    type: blob.type || 'image/jpeg',
    lastModified: Date.now()
  });
};


const handleDrop = (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file && file.type.match('image.*')) {
    processImageFile(file);
  }
};

const processImageFile = (file) => {
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.value = e.target.result;
    emit('file-upload', file);
  };
  reader.readAsDataURL(file);
};

const clearImage = () => {
  imagePreview.value = null;
  emit('clear');
};
</script>

<style scoped>
.upload-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.upload-box {
  border: 2px dashed #cbd5e1;
  border-radius: 12px;
  padding: 3rem 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  margin-bottom: 1.5rem;
  position: relative;
}

.upload-box:hover {
  border-color: #93c5fd;
  background: #f8fafc;
}

.upload-box.has-image {
  padding: 0;
  border: 1px solid #e2e8f0;
}

.upload-icon {
  color: #93c5fd;
  margin-bottom: 1rem;
}

.upload-icon svg {
  display: inline-block;
}

.upload-hint {
  font-size: 0.875rem;
  color: #94a3b8;
  margin-top: 0.5rem;
}

.image-preview {
  width: 100%;
  height: 100%;
  position: relative;
}

.image-preview img {
  width: 100%;
  max-height: 400px;
  object-fit: contain;
  border-radius: 12px;
}

.clear-btn {
  position: absolute;
  top: 1rem;
  right: 1rem;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
}

.clear-btn:hover {
  background: rgba(0, 0, 0, 0.8);
}

.sample-images p {
  font-size: 0.875rem;
  color: #64748b;
  margin-bottom: 0.75rem;
}

.sample-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 1rem;
}

.sample-item {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  aspect-ratio: 1;
  transition: all 0.2s;
}

.sample-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.sample-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.sample-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 0.5rem;
  font-size: 0.75rem;
  text-align: center;
}
</style>
