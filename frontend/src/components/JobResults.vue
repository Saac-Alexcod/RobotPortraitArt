<script setup lang="ts">
import { computed } from 'vue'

import { getJobFileUrl } from '../api/client'
import DrawingSimulation from './DrawingSimulation.vue'
import ResultComparison from './ResultComparison.vue'
import type {
  JobStatusResponse,
  ManifestResponse,
} from '../api/types'

const props = defineProps<{
  job: JobStatusResponse
  manifest: ManifestResponse | null
  manifestError?: string | null
}>()

const resultCards = computed(() => {
  const files = props.job.result_files
  const cards = [
    {
      title: 'Original',
      subtitle: props.job.original_filename || props.job.input_file,
      filename: props.job.input_file,
      previewKind: 'photo',
    },
    {
      title: files.aligned_input ? 'Pipeline Prepared' : 'Prepared',
      subtitle: files.prepared || 'prepared image',
      filename: files.prepared,
      previewKind: 'photo',
    },
  ]

  if (files.aligned_input) {
    cards.push({
      title: 'Aligned Input',
      subtitle: files.aligned_input,
      filename: files.aligned_input,
      previewKind: 'photo',
    })
  }

  cards.push(
    {
      title: 'Portrait',
      subtitle: files.raster_portrait || `${props.job.method} raster`,
      filename: files.raster_portrait,
      previewKind: 'portrait',
    },
    {
      title: 'Robot Vector',
      subtitle: files.vector_svg || files.vector_render || 'vector result',
      filename: files.vector_render || files.vector_svg,
      previewKind: 'vector',
    },
  )

  return cards
})

const vectorMetrics = computed(
  () => props.manifest?.vectorization,
)

const timings = computed(
  () => props.manifest?.timings_ms,
)

const comparisonOptions = computed(() => {
  const files = props.job.result_files

  return [
    {
      key: 'original',
      label: 'Original',
      filename: props.job.input_file,
      url: fileUrl(props.job.input_file),
    },
    {
      key: 'prepared',
      label: files.aligned_input ? 'Pipeline Prepared' : 'Prepared',
      filename: files.prepared || '',
      url: fileUrl(files.prepared),
    },
    {
      key: 'aligned',
      label: 'Aligned Input',
      filename: files.aligned_input || '',
      url: fileUrl(files.aligned_input),
    },
    {
      key: 'portrait',
      label: 'Portrait',
      filename: files.raster_portrait || '',
      url: fileUrl(files.raster_portrait),
    },
    {
      key: 'vector',
      label: 'Robot Vector',
      filename: files.vector_render || files.vector_svg || '',
      url: fileUrl(files.vector_render || files.vector_svg),
    },
  ]
})

const optimization = computed(() => {
  const ratio = vectorMetrics.value?.two_opt_improvement_ratio
  return typeof ratio === 'number' && Number.isFinite(ratio)
    ? `${(ratio * 100).toFixed(2)}%`
    : null
})

const strokeCount = computed(
  () => (
    vectorMetrics.value?.stroke_count
    ?? vectorMetrics.value?.path_count
  ),
)

function fileUrl(filename: string | undefined): string {
  if (!filename) {
    return ''
  }

  return getJobFileUrl(
    props.job.job_id,
    filename,
  )
}

function formatNumber(
  value: number | undefined,
  digits = 0,
): string {
  if (typeof value !== 'number') {
    return '—'
  }

  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

function formatMilliseconds(
  value: number | undefined,
): string {
  if (typeof value !== 'number') {
    return '—'
  }

  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)} s`
  }

  return `${value.toFixed(1)} ms`
}
</script>

<template>
  <div class="results-shell">
    <div class="section-heading results-heading">
      <div>
        <span>RESULTS</span>
        <h2>生成结果</h2>
      </div>

      <div class="completed-chip">
        <span></span>
        Completed
      </div>
    </div>

    <div
      class="result-grid"
      :class="{
        'result-grid--expanded': resultCards.length > 4,
      }"
    >
      <article
        v-for="result in resultCards"
        :key="result.title"
        class="result-card"
      >
        <div class="result-card__media">
          <img
            v-if="result.filename"
            :src="fileUrl(result.filename)"
            :alt="`${result.title} result`"
            :class="{
              'result-card__image--vector':
                result.previewKind === 'vector',
            }"
          >
          <div v-else class="result-card__empty">
            No file
          </div>
        </div>

        <div class="result-card__caption">
          <strong>{{ result.title }}</strong>
          <span>{{ result.subtitle }}</span>
        </div>
      </article>
    </div>

    <div
      v-if="job.result_files.aligned_input"
      class="aligned-input-note"
    >
      <strong>Aligned Input</strong>
      <span>
        YuNet 五点检测与 512×512 similarity alignment 后，
        实际送入 APDrawingGAN 的图像。
      </span>
    </div>

    <div class="metrics-grid">
      <article>
        <span>绘制笔画数 · Strokes</span>
        <strong>{{ formatNumber(strokeCount) }}</strong>
      </article>

      <article>
        <span>落笔长度 · Pen-down</span>
        <strong>
          {{ formatNumber(vectorMetrics?.pen_down_length_px, 1) }}
          <small>px</small>
        </strong>
      </article>

      <article>
        <span>抬笔空程 · Pen-up</span>
        <strong>
          {{ formatNumber(vectorMetrics?.pen_up_length_px, 1) }}
          <small>px</small>
        </strong>
      </article>

      <article>
        <span>矢量化耗时</span>
        <strong>{{ formatMilliseconds(timings?.vectorize) }}</strong>
      </article>

      <article>
        <span>任务总耗时</span>
        <strong>{{ formatMilliseconds(timings?.total) }}</strong>
      </article>

      <article v-if="optimization !== null" class="metric-optimization">
        <span>2-opt 空程优化率</span>
        <strong>↓ {{ optimization }}</strong>
        <small v-if="typeof vectorMetrics?.pen_up_greedy_px === 'number'">
          {{ formatNumber(vectorMetrics.pen_up_greedy_px, 1) }} →
          {{ formatNumber(vectorMetrics.pen_up_length_px, 1) }} px
        </small>
      </article>
    </div>

    <p class="metrics-explainer">
      落笔长度表示画线行程；抬笔空程表示笔尖离纸后的移动距离。优化率对比 Greedy 初始排序与 2-opt 排序，不代表笔画质量提升。
    </p>

    <div v-if="!manifest" class="manifest-loading">
      {{ manifestError || '任务已完成，正在读取 Manifest 指标…' }}
    </div>

    <ResultComparison
      :options="comparisonOptions"
    />

    <DrawingSimulation
      v-if="job.result_files.vector_svg"
      :job-id="job.job_id"
      :filename="job.result_files.vector_svg"
    />

    <div class="download-row">
      <span>DOWNLOAD</span>

      <a
        v-if="job.result_files.raster_portrait"
        :href="fileUrl(job.result_files.raster_portrait)"
        :download="job.result_files.raster_portrait"
      >
        Portrait PNG
      </a>

      <a
        v-if="job.result_files.vector_svg"
        :href="fileUrl(job.result_files.vector_svg)"
        :download="job.result_files.vector_svg"
      >
        Robot SVG
      </a>

      <a
        v-if="job.result_files.manifest"
        :href="fileUrl(job.result_files.manifest)"
        :download="job.result_files.manifest"
      >
        Manifest JSON
      </a>
    </div>

    <div class="result-footnote">
      <code>{{ job.job_id }}</code>
      <span>{{ job.method }} · {{ job.vector_mode }} · {{ job.size }} px</span>
    </div>
  </div>
</template>
