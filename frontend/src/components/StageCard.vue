<script setup lang="ts">
defineProps<{
  step: string
  title: string
  description: string
  status: 'ready' | 'next'
}>()
</script>

<template>
  <article class="stage-card">
    <div class="stage-card__top">
      <span class="stage-card__step">{{ step }}</span>
      <span
        class="stage-card__status"
        :class="`stage-card__status--${status}`"
      >
        {{ status === 'ready' ? 'Ready' : 'Next' }}
      </span>
    </div>

    <h3>{{ title }}</h3>
    <p>{{ description }}</p>
  </article>
</template>
