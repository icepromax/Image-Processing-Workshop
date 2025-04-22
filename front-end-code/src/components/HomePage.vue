
<template>
  <main class="main-content">
      <IntroSection/>
      <DemoSection/>
  </main>
</template>

<script setup>
  import DemoSection from "@/components/DemoSection.vue";
  import IntroSection from "@/components/IntroSection.vue";


</script>


<style scoped>
.main-content {
  flex: 1;
  padding: 2rem 1rem;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}
</style>
