<script setup lang="ts">
import { computed } from 'vue'

import { getJobFileUrl } from '../api/client'
import DrawingSimulation from './DrawingSimulation.vue'
import type {
  JobStatusResponse,
  ManifestResponse,
} from '../api/types'

const props = defineProps<{
  job: JobStatusResponse
  manifest: ManifestResponse | null
}>()

const resultCards = computed(() => {
  const files = props.job.result_files

  return [
    {
      title: 'Original',
      subtitle: props.job.original_filename || props.job.input_file,
      filename: props.job.input_file,
    },
    {
      title: 'Prepared',
      subtitle: files.prepared || 'prepared image',
      filename: files.prepared,
    },
    {
      title: 'Portrait',
      subtitle: files.raster_portrait || `${props.job.method} raster`,
      filename: files.raster_portrait,
    },
    {
      title: 'Robot Vector',
      subtitle: files.vector_svg || files.vector_render || 'vector result',
      filename: files.vector_svg || files.vector_render,
    },
  ]
})

const vectorMetrics = computed(
  () => props.manifest?.vectorization,
)

const timings = computed(
  () => props.manifest?.timings_ms,
)

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

    <div class="result-grid">
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

    <div class="metrics-grid">
      <article>
        <span>Stroke Count</span>
        <strong>{{ formatNumber(strokeCount) }}</strong>
      </article>

      <article>
        <span>Pen-down</span>
        <strong>
          {{ formatNumber(vectorMetrics?.pen_down_length_px, 1) }}
          <small>px</small>
        </strong>
      </article>

      <article>
        <span>Pen-up</span>
        <strong>
          {{ formatNumber(vectorMetrics?.pen_up_length_px, 1) }}
          <small>px</small>
        </strong>
      </article>

      <article>
        <span>Vectorize</span>
        <strong>{{ formatMilliseconds(timings?.vectorize) }}</strong>
      </article>

      <article>
        <span>Total Time</span>
        <strong>{{ formatMilliseconds(timings?.total) }}</strong>
      </article>
    </div>

    <div v-if="!manifest" class="manifest-loading">
      Job 已完成，正在读取 manifest 指标…
    </div>

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
