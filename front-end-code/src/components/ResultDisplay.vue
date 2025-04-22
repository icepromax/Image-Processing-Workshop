<template>
  <div class="result-section">
    <h3 class="result-title">结果对比</h3>
    <div class="comparison-view">
      <div class="image-column">
        <h4>原始图片</h4>
        <div class="image-container">
          <img :src="originalImage" alt="原始图片">
        </div>
      </div>
      <div class="image-column">
        <h4>转换结果</h4>
        <div class="image-container">
          <img :src="repairedImage" alt="修复图片">
          <div class="result-actions">
            <button @click="$emit('download')" class="download-btn">
              <svg viewBox="0 0 24 24" width="16" height="16">
                <path d="M5,20H19V18H5M19,9H15V3H9V9H5L12,16L19,9Z" fill="currentColor"/>
              </svg>
              下载
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  originalImage: {
    type: String,
    required: true
  },
  repairedImage: {
    type: String,
    required: true
  }
});

defineEmits(['download', 'reset']);
</script>

<style scoped>
.result-section {
  background: white;
  border-radius: 16px;
  padding: 2.5rem;
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

.result-title {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 2rem;
  text-align: center;
  color: #1e293b;
}

.comparison-view {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

.image-column {
  display: flex;
  flex-direction: column;
}

.image-column h4 {
  font-weight: 600;
  margin-bottom: 1rem;
  color: #475569;
  text-align: center;
}

.image-container {
  flex: 1;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
}

.image-container img {
  width: 100%;
  max-height: 400px;
  object-fit: contain;
  border-radius: 8px;
  margin-bottom: 1rem;
}

.result-actions {
  display: flex;
  gap: 0.75rem;
  margin-top: auto;
}

.download-btn, .new-btn {
  flex: 1;
  padding: 0.75rem;
  border-radius: 8px;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  cursor: pointer;
  transition: all 0.2s;
}

.download-btn {
  background: #e0f2fe;
  color: #0369a1;
  border: 1px solid #bae6fd;
}

.download-btn:hover {
  background: #bae6fd;
}

.new-btn {
  background: #f1f5f9;
  color: #475569;
  border: 1px solid #e2e8f0;
}

.new-btn:hover {
  background: #e2e8f0;
}

@media (max-width: 768px) {
  .comparison-view {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
}
</style>
