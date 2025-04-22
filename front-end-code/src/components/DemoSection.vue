<template>
  <section class="demo-section">
    <div class="demo-card">
      <h2 class="section-title">开始改造您的图片</h2>

      <ModelSelector
          :models="models"
          @update:model="selectedModel = $event"
      />

      <ImageUploader
          :samples="samples"
          @file-upload="handleFileUpload"
          @select-sample="handleSampleSelect"
          @clear="reset"
      />

      <div class="action-buttons">
        <button
            class="repair-button"
            :disabled="!originalImage || isProcessing"
            @click="repairImage1"
        >
          <span v-if="!isProcessing">开始改造</span>
          <span v-else class="processing">
            <span class="spinner"></span>
            修复中...
          </span>
        </button>
      </div>
    </div>

    <ResultDisplay
        v-if="repairedImage"
        :original-image="originalImage"
        :repaired-image="repairedImage"
        @download="downloadResult"
        @reset="reset"
    />
  </section>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { repairImage, getSampleImages, getAvailableModels } from '@/api/imageRepair';
import ModelSelector from '@/components/common/ModelSelector.vue';
import ImageUploader from '@/components/common/ImageUploader.vue';
import ResultDisplay from '@/components/ResultDisplay.vue';

const models = ref([]);
const samples = ref([]);
const selectedModel = ref('');
const originalImage = ref(null);
const originalFile = ref(null);
const repairedImage = ref(null);
const isProcessing = ref(false);

onMounted(async () => {
  models.value = await getAvailableModels();
  samples.value = await getSampleImages();
  selectedModel.value = models.value[0]?.id;
});

const handleFileUpload = (file) => {
  originalFile.value = file;
  originalImage.value = URL.createObjectURL(file); // 生成预览图
};

const handleSampleSelect = (sample) => {
  originalImage.value = sample.full;
  originalFile.value = null; // 示例图片没有File对象
};

const repairImage1 = async () => {
  if (!originalImage.value || isProcessing.value) return;

  isProcessing.value = true;

  try {
    let fileToRepair;

    if (originalFile.value) {
      fileToRepair = originalFile.value;
    } else {
      // 如果是示例图片，从Base64转换为Blob
      const response = await fetch(originalImage.value);
      fileToRepair = await response.blob();
    }

    repairedImage.value = await repairImage(fileToRepair, selectedModel.value);
  } catch (error) {
    console.error('修复失败:', error.message);
    alert(error.message);
  } finally {
    isProcessing.value = false;
  }
};

const reset = () => {
  originalImage.value = null;
  originalFile.value = null;
  repairedImage.value = null;
};

const downloadResult = () => {
  if (!repairedImage.value) return;

  const link = document.createElement('a');
  link.href = repairedImage.value;
  link.download = `repaired-${Date.now()}.jpg`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

  const openLink = (url) => {
  window.open(url, '_blank');
  };
</script>

<style scoped>
.demo-section {
  margin-bottom: 3rem;
}

.demo-card {
  background: white;
  border-radius: 16px;
  padding: 2.5rem;
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
  margin-bottom: 3rem;
}

.section-title {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 2rem;
  text-align: center;
  color: #1e293b;
}

.action-buttons {
  display: flex;
  justify-content: center;
  margin-top: 2rem;
}

.repair-button {
  padding: 0.875rem 2rem;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2);
}

.repair-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 6px 8px rgba(99, 102, 241, 0.3);
}

.repair-button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.processing {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
