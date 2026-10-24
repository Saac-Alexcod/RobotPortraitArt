<script setup lang="ts">
import {
  computed,
  ref,
  watch,
} from 'vue'

export interface ComparisonOption {
  key: string
  label: string
  filename: string
  url: string
}

const props = defineProps<{
  options: ComparisonOption[]
}>()

const leftKey = ref('')
const rightKey = ref('')
const split = ref(50)

const validOptions = computed(
  () => props.options.filter(
    (option) => option.filename && option.url,
  ),
)

function chooseDefaults(): void {
  const options = validOptions.value

  if (options.length === 0) {
    leftKey.value = ''
    rightKey.value = ''
    return
  }

  const original = options.find((option) => option.key === 'original')
  const aligned = options.find((option) => option.key === 'aligned')
  const portrait = options.find((option) => option.key === 'portrait')
  const vector = options.find((option) => option.key === 'vector')

  // Compare the actual GAN input with its output by default.  Users can
  // switch to Portrait / Robot Vector to inspect the vectorization stage.
  leftKey.value = aligned?.key ?? original?.key ?? options[0]?.key ?? ''
  rightKey.value = portrait?.key ?? vector?.key ?? options[1]?.key ?? leftKey.value

  if (
    rightKey.value === leftKey.value
    && options.length > 1
  ) {
    rightKey.value = (
      options.find(
        (option) => option.key !== leftKey.value,
      )?.key
      ?? rightKey.value
    )
  }
}

watch(
  validOptions,
  () => {
    const keys = new Set(
      validOptions.value.map(
        (option) => option.key,
      ),
    )

    if (
      !keys.has(leftKey.value)
      || !keys.has(rightKey.value)
    ) {
      chooseDefaults()
    }
  },
  {
    immediate: true,
  },
)

const left = computed(
  () => validOptions.value.find(
    (option) => option.key === leftKey.value,
  ) ?? null,
)

const right = computed(
  () => validOptions.value.find(
    (option) => option.key === rightKey.value,
  ) ?? null,
)

function setLeft(key: string): void {
  leftKey.value = key

  if (rightKey.value === key) {
    rightKey.value = (
      validOptions.value.find(
        (option) => option.key !== key,
      )?.key
      ?? rightKey.value
    )
  }
}

function setRight(key: string): void {
  rightKey.value = key

  if (leftKey.value === key) {
    leftKey.value = (
      validOptions.value.find(
        (option) => option.key !== key,
      )?.key
      ?? leftKey.value
    )
  }
}
</script>

<template>
  <section
    v-if="left && right"
    class="result-comparison"
  >
    <div class="result-comparison__header">
      <div>
        <span>A/B COMPARISON</span>
        <h3>结果叠加对比</h3>
        <p>
          拖动分割线，对比原图、预处理、线稿和机器人矢量结果。
        </p>
      </div>

      <div class="comparison-value">
        {{ split }}%
      </div>
    </div>

    <div class="comparison-pickers">
      <div class="comparison-picker">
        <span>LEFT</span>

        <div class="comparison-picker__options">
          <button
            v-for="option in validOptions"
            :key="`left-${option.key}`"
            type="button"
            :class="{
              'comparison-picker__active':
                option.key === leftKey,
            }"
            @click="setLeft(option.key)"
          >
            {{ option.label }}
          </button>
        </div>
      </div>

      <div class="comparison-picker">
        <span>RIGHT</span>

        <div class="comparison-picker__options">
          <button
            v-for="option in validOptions"
            :key="`right-${option.key}`"
            type="button"
            :class="{
              'comparison-picker__active':
                option.key === rightKey,
            }"
            @click="setRight(option.key)"
          >
            {{ option.label }}
          </button>
        </div>
      </div>
    </div>

    <div class="comparison-stage">
      <img
        class="comparison-image"
        :src="right.url"
        :alt="right.label"
      >

      <img
        class="comparison-image comparison-image--overlay"
        :src="left.url"
        :alt="left.label"
        :style="{
          clipPath: `inset(0 ${100 - split}% 0 0)`,
        }"
      >

      <div
        class="comparison-divider"
        :style="{ left: `${split}%` }"
      >
        <span></span>
      </div>

      <div class="comparison-label comparison-label--left">
        {{ left.label }}
      </div>

      <div class="comparison-label comparison-label--right">
        {{ right.label }}
      </div>

      <input
        v-model.number="split"
        class="comparison-range"
        type="range"
        min="0"
        max="100"
        step="1"
        aria-label="A/B comparison split"
      >
    </div>

    <div class="comparison-caption">
      <span>
        ← {{ left.label }}
      </span>
      <span>
        {{ right.label }} →
      </span>
    </div>
  </section>
</template>
