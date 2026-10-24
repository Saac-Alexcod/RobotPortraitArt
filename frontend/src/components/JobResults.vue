<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from 'vue'

import { getJobFileUrl } from '../api/client'
import DrawingSimulation from './DrawingSimulation.vue'
import ResultComparison from './ResultComparison.vue'
import type { JobStatusResponse, ManifestResponse } from '../api/types'

const props = defineProps<{
  job: JobStatusResponse
  manifest: ManifestResponse | null
  manifestError?: string | null
}>()

interface ResultStage {
  key: string
  title: string
  english: string
  number: string
  description: string
  filename?: string
  kind: 'photo' | 'portrait' | 'vector'
}

const resultStages = computed<ResultStage[]>(() => {
  const files = props.job.result_files
  let step = 0
  const stage = (
    key: string,
    title: string,
    english: string,
    description: string,
    filename: string | undefined,
    kind: ResultStage['kind'],
  ): ResultStage => ({
    key,
    title,
    english,
    description,
    filename,
    kind,
    number: String(++step).padStart(2, '0'),
  })

  const stages = [
    stage('original', '原始照片', 'Original', '用户上传的原始输入图像', props.job.input_file, 'photo'),
    stage('prepared', '预处理', 'Pipeline Prepared', '图像预处理后的输出', files.prepared, 'photo'),
  ]

  if (files.aligned_input) {
    stages.push(stage('aligned', '对齐输入', 'Aligned Input', '实际送入 APDrawingGAN 的图像', files.aligned_input, 'photo'))
  }

  stages.push(
    stage('portrait', '艺术肖像', 'GAN Portrait', '模型生成的栅格线稿', files.raster_portrait, 'portrait'),
    stage('vector', '机器人轨迹', 'Robot Vector', '可下载的 SVG 矢量路径', files.vector_render || files.vector_svg, 'vector'),
  )
  return stages
})

const featuredStages = computed(() => resultStages.value.filter(
  (stage) => stage.kind === 'portrait' || stage.kind === 'vector',
))
const inputStages = computed(() => resultStages.value.filter(
  (stage) => stage.kind === 'photo',
))
type ResultTab = 'artwork' | 'drawing' | 'comparison' | 'analysis'
const activeTab = ref<ResultTab>('artwork')
// Full image is the default; close-up is a deliberate, reversible viewing choice.
const artworkView = ref<'full' | 'detail'>('full')
function activateTab(tab: ResultTab): void {
  activeTab.value = tab
}

const activePreview = ref<ResultStage | null>(null)
const closePreviewButton = ref<HTMLButtonElement | null>(null)
const lightboxPanel = ref<HTMLDivElement | null>(null)
let previousFocus: HTMLElement | null = null

const previewableStages = computed(() => resultStages.value.filter((stage) => Boolean(stage.filename)))
const previewPosition = computed(() => previewableStages.value.findIndex((stage) => stage.key === activePreview.value?.key))
function movePreview(direction: 1 | -1): void {
  const all = previewableStages.value
  if (!activePreview.value || all.length < 2) return
  const position = previewPosition.value
  const next = (position + direction + all.length) % all.length
  activePreview.value = all[next] ?? null
}

function fileUrl(filename: string | undefined): string {
  return filename ? getJobFileUrl(props.job.job_id, filename) : ''
}

function showPreview(stage: ResultStage): void {
  if (stage.filename) {
    previousFocus = document.activeElement instanceof HTMLElement
      ? document.activeElement : null
    activePreview.value = stage
    void nextTick(() => closePreviewButton.value?.focus())
  }
}

function closePreview(): void {
  if (!activePreview.value) return
  activePreview.value = null
  void nextTick(() => previousFocus?.focus())
}

function handleEscape(event: KeyboardEvent): void {
  if (!activePreview.value) return
  if (event.key === 'Escape') {
    event.preventDefault()
    closePreview()
    return
  }
  if (event.key === 'ArrowLeft') {
    event.preventDefault()
    movePreview(-1)
  } else if (event.key === 'ArrowRight') {
    event.preventDefault()
    movePreview(1)
  } else if (event.key === 'Tab' && lightboxPanel.value) {
    const focusable = Array.from(lightboxPanel.value.querySelectorAll<HTMLElement>('button:not(:disabled), a[href]'))
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (!first || !last) return
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }
}

watch(() => props.job.job_id, () => {
  closePreview()
  activeTab.value = 'artwork'
  artworkView.value = 'full'
})
onMounted(() => window.addEventListener('keydown', handleEscape))
onBeforeUnmount(() => window.removeEventListener('keydown', handleEscape))

const vectorMetrics = computed(() => props.manifest?.vectorization)
const timings = computed(() => props.manifest?.timings_ms)
const strokeCount = computed(() => vectorMetrics.value?.stroke_count ?? vectorMetrics.value?.path_count)
const optimizationRatio = computed(() => {
  const ratio = vectorMetrics.value?.two_opt_improvement_ratio
  return typeof ratio === 'number' && Number.isFinite(ratio) ? ratio : null
})
const greedyDistance = computed(() => vectorMetrics.value?.pen_up_greedy_px)
const optimizedDistance = computed(() => vectorMetrics.value?.pen_up_length_px)
const showPathComparison = computed(() => (
  typeof greedyDistance.value === 'number'
  && Number.isFinite(greedyDistance.value)
  && greedyDistance.value > 0
  && typeof optimizedDistance.value === 'number'
  && Number.isFinite(optimizedDistance.value)
  && optimizedDistance.value >= 0
))
const savedDistance = computed(() => {
  if (!showPathComparison.value) return undefined
  return greedyDistance.value! - optimizedDistance.value!
})
const optimizedBarWidth = computed(() => {
  if (!showPathComparison.value) return '0%'
  return `${Math.min(100, Math.max(0, (optimizedDistance.value! / greedyDistance.value!) * 100))}%`
})

const comparisonOptions = computed(() => {
  const files = props.job.result_files
  return [
    { key: 'original', label: 'Original', filename: props.job.input_file, url: fileUrl(props.job.input_file) },
    { key: 'prepared', label: 'Prepared', filename: files.prepared || '', url: fileUrl(files.prepared) },
    { key: 'aligned', label: 'Aligned Input', filename: files.aligned_input || '', url: fileUrl(files.aligned_input) },
    { key: 'portrait', label: 'Portrait', filename: files.raster_portrait || '', url: fileUrl(files.raster_portrait) },
    { key: 'vector', label: 'Robot Vector', filename: files.vector_render || files.vector_svg || '', url: fileUrl(files.vector_render || files.vector_svg) },
  ]
})

function formatNumber(value: number | undefined, digits = 0): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
    : '—'
}

function formatMilliseconds(value: number | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(1)} ms`
}
</script>

<template>
  <section class="results-shell studio-results" id="generation-results" aria-labelledby="results-title">
    <div class="studio-results__heading">
      <div><span class="studio-eyebrow">YOUR CREATION · {{ job.method.toUpperCase() }}</span><h2 id="results-title">你的创作结果</h2><p>查看艺术肖像、逐笔绘制与算法分析，一切来自此次真实生成任务。</p></div>
      <div class="studio-results__complete"><span class="status-dot"></span> 已完成</div>
    </div>
    <nav class="studio-result-tabs" aria-label="结果查看方式">
      <button type="button" :class="{ 'studio-result-tabs__active': activeTab === 'artwork' }" :aria-pressed="activeTab === 'artwork'" @click="activateTab('artwork')"><span aria-hidden="true">◫</span> 作品</button>
      <button type="button" :class="{ 'studio-result-tabs__active': activeTab === 'drawing' }" :aria-pressed="activeTab === 'drawing'" @click="activateTab('drawing')"><span aria-hidden="true">▷</span> 逐笔绘制</button>
      <button type="button" :class="{ 'studio-result-tabs__active': activeTab === 'comparison' }" :aria-pressed="activeTab === 'comparison'" @click="activateTab('comparison')"><span aria-hidden="true">⇆</span> 图像对比</button>
      <button type="button" :class="{ 'studio-result-tabs__active': activeTab === 'analysis' }" :aria-pressed="activeTab === 'analysis'" @click="activateTab('analysis')"><span aria-hidden="true">▥</span> 技术数据</button>
    </nav>

    <div v-if="activeTab === 'artwork'" class="studio-tab-panel" id="studio-artwork">
      <div class="studio-artwork-actions">
        <div class="studio-artwork-actions__intro">
          <strong>艺术肖像 · 机器人轨迹</strong>
          <p>{{ artworkView === 'full' ? '完整呈现两张真实输出，不裁切画面；点击可放大。' : '局部观察模式仅改变浏览器裁切方式，不修改任何生成文件。' }}</p>
        </div>
        <div class="studio-artwork-actions__controls">
          <div class="studio-artwork-view" role="group" aria-label="图像显示方式">
            <button type="button" :class="{ 'studio-artwork-view__selected': artworkView === 'full' }" :aria-pressed="artworkView === 'full'" @click="artworkView = 'full'">完整画幅</button>
            <button type="button" :class="{ 'studio-artwork-view__selected': artworkView === 'detail' }" :aria-pressed="artworkView === 'detail'" @click="artworkView = 'detail'">观察细节</button>
          </div>
          <button class="studio-artwork-actions__drawing" type="button" @click="activateTab('drawing')">逐笔绘制 <span aria-hidden="true">→</span></button>
        </div>
      </div>
    <section class="result-featured" aria-label="核心生成成果">
      <div class="result-featured__title">
        <div>
          <span class="result-kicker">01 / 核心成果</span>
          <h3>两种形式，一条完整绘制链路</h3>
        </div>
        <p>左侧呈现 GAN 生成的艺术线稿，右侧呈现实际导出的机器人 SVG 轨迹。</p>
      </div>
      <div class="result-feature-grid" :class="{ 'result-feature-grid--detail': artworkView === 'detail' }">
        <article v-for="stage in featuredStages" :key="stage.key" class="result-feature-card" :class="`result-feature-card--${stage.kind}`">
          <div class="result-feature-card__top">
            <div class="result-feature-card__identity">
              <span class="result-feature-card__number">{{ stage.number }}</span>
              <div><strong>{{ stage.title }}</strong><small>{{ stage.english }}</small></div>
            </div>
            <span class="result-feature-card__type">{{ stage.kind === 'vector' ? 'SVG PATH' : 'RASTER' }}</span>
          </div>
          <button
            v-if="stage.filename"
            type="button"
            class="result-feature-card__media"
            :aria-label="`放大查看${stage.title}`"
            @click="showPreview(stage)"
          >
            <img :src="fileUrl(stage.filename)" :alt="stage.description" loading="lazy">
            <span class="result-feature-card__zoom">放大查看 ↗</span>
          </button>
          <div v-else class="result-feature-card__missing">此阶段暂无可预览文件</div>
          <div class="result-feature-card__bottom">
            <span>{{ stage.description }}</span>
            <code>{{ stage.filename || '—' }}</code>
          </div>
        </article>
      </div>
    </section>


      <div class="studio-highlight-metrics" aria-label="关键生成指标">
        <div><span>路径笔画</span><strong>{{ formatNumber(strokeCount) }}</strong><small>Strokes</small></div>
        <div><span>空程优化</span><strong v-if="optimizationRatio !== null">{{ (optimizationRatio * 100).toFixed(2) }}%</strong><strong v-else>—</strong><small>Greedy → 2-opt</small></div>
        <div><span>总耗时</span><strong>{{ formatMilliseconds(timings?.total) }}</strong><small>Pipeline</small></div>
      </div>
      <details class="studio-stage-details"><summary><span>查看原始照片与处理过程</span><span aria-hidden="true">⌄</span></summary>
    <section class="result-inputs" aria-label="处理阶段回溯">
      <div class="result-featured__title">
        <div>
          <span class="result-kicker">02 / 输入过程</span>
          <h3>完整处理阶段可回溯</h3>
        </div>
        <p>所有缩略图均直接读取该次任务产生的真实文件。</p>
      </div>
      <div class="result-input-grid" :class="{ 'result-input-grid--two': inputStages.length === 2 }">
        <article v-for="stage in inputStages" :key="stage.key" class="result-input-card">
          <button
            v-if="stage.filename"
            type="button"
            class="result-input-card__media"
            :aria-label="`放大查看${stage.title}`"
            @click="showPreview(stage)"
          >
            <img :src="fileUrl(stage.filename)" :alt="stage.description" loading="lazy">
          </button>
          <div v-else class="result-input-card__missing">暂无预览</div>
          <div class="result-input-card__text">
            <span>{{ stage.number }} · {{ stage.english }}</span>
            <strong>{{ stage.title }}</strong>
            <small>{{ stage.description }}</small>
          </div>
        </article>
      </div>
      <p v-if="job.result_files.aligned_input" class="result-inputs__note">
        <strong>Aligned Input 说明：</strong>经 YuNet 五点检测与 512 × 512 相似变换对齐后，实际输入 APDrawingGAN 的图像。它不同于原始照片和 Pipeline Prepared。
      </p>
    </section>


      </details>
    </div>

    <div v-else-if="activeTab === 'drawing'" class="studio-tab-panel studio-tab-panel--drawing">
      <div class="studio-tab-intro"><strong>每一条线都有自己的绘制顺序</strong><p>使用后端生成的 SVG Path，按优化后的路径顺序模拟落笔与抬笔移动。</p></div>
      <DrawingSimulation v-if="job.result_files.vector_svg" :job-id="job.job_id" :filename="job.result_files.vector_svg" />
      <p v-else class="manifest-loading">当前任务没有可播放的 SVG 轨迹。</p>
    </div>

    <div v-else-if="activeTab === 'comparison'" class="studio-tab-panel studio-tab-panel--comparison">
      <div class="studio-tab-intro"><strong>把每一次变化放在同一张画布上</strong><p>拖动中线，比较模型输入与输出，或艺术肖像与机器人轨迹的差异。</p></div>
      <ResultComparison :options="comparisonOptions" />
    </div>

    <div v-else class="studio-tab-panel studio-tab-panel--analysis">
    <section class="result-performance" aria-label="运行结果与轨迹指标">
      <div class="result-featured__title">
        <div>
          <span class="result-kicker">03 / 可量化结果</span>
          <h3>本次任务的真实指标</h3>
        </div>
        <p>以下数字读取 Manifest；像素距离为图像坐标尺度，并非实际机械臂行程。</p>
      </div>
      <div class="metrics-grid metrics-grid--presentation">
        <article><span>绘制笔画数 · Strokes</span><strong>{{ formatNumber(strokeCount) }}</strong></article>
        <article><span>落笔长度 · Pen-down</span><strong>{{ formatNumber(vectorMetrics?.pen_down_length_px, 1) }} <small>px</small></strong></article>
        <article><span>抬笔空程 · Pen-up</span><strong>{{ formatNumber(vectorMetrics?.pen_up_length_px, 1) }} <small>px</small></strong></article>
        <article><span>矢量化耗时</span><strong>{{ formatMilliseconds(timings?.vectorize) }}</strong></article>
        <article><span>任务总耗时</span><strong>{{ formatMilliseconds(timings?.total) }}</strong></article>
        <article v-if="optimizationRatio !== null" class="metric-optimization"><span>2-opt 抬笔空程优化率</span><strong>{{ optimizationRatio >= 0 ? '↓' : '↑' }} {{ Math.abs(optimizationRatio * 100).toFixed(2) }}%</strong><small>相对于 Greedy 初始排序</small></article>
      </div>

      <div v-if="showPathComparison" class="optimizer-story">
        <div class="optimizer-story__intro">
          <div><strong>路径规划优化对比</strong><span>Greedy 与 2-opt 的空程距离</span></div>
          <span class="optimizer-story__saving" v-if="savedDistance !== undefined">{{ savedDistance >= 0 ? '减少' : '增加' }} {{ formatNumber(Math.abs(savedDistance), 1) }} px</span>
        </div>
        <div class="optimizer-story__track">
          <div class="optimizer-story__label"><span>Greedy 初始路径</span><strong>{{ formatNumber(greedyDistance, 1) }} px</strong></div>
          <div class="optimizer-story__bar optimizer-story__bar--greedy"><span style="width:100%"></span></div>
        </div>
        <div class="optimizer-story__track">
          <div class="optimizer-story__label"><span>2-opt 优化路径</span><strong>{{ formatNumber(optimizedDistance, 1) }} px</strong></div>
          <div class="optimizer-story__bar optimizer-story__bar--optimized"><span :style="{ width: optimizedBarWidth }"></span></div>
        </div>
        <p>2-opt 优化的是笔画之间的抬笔移动顺序，不意味着线稿画质提升或实体机器人时间等比例缩短。</p>
      </div>
      <p v-else class="metrics-explainer">落笔长度表示实际画线距离；抬笔空程表示笔尖离纸后的移动距离。未提供可比较的路径规划指标时不展示优化图表。</p>
      <div v-if="!manifest" class="manifest-loading" role="status">{{ manifestError || '任务已完成，正在读取 Manifest 指标…' }}</div>
    </section>


    </div>

    <div class="download-row">
      <span>RESULT FILES</span>
      <a v-if="job.result_files.raster_portrait" :href="fileUrl(job.result_files.raster_portrait)" :download="job.result_files.raster_portrait">肖像 PNG ↓</a>
      <a v-if="job.result_files.vector_svg" :href="fileUrl(job.result_files.vector_svg)" :download="job.result_files.vector_svg">机器人 SVG ↓</a>
      <a v-if="job.result_files.manifest" :href="fileUrl(job.result_files.manifest)" :download="job.result_files.manifest">Manifest JSON ↓</a>
    </div>
    <div class="result-footnote"><code>{{ job.job_id }}</code><span>{{ job.method }} · {{ job.vector_mode }} · {{ job.size }} px</span></div>

    <div v-if="activePreview" class="result-lightbox" role="dialog" aria-modal="true" aria-keyshortcuts="Escape ArrowLeft ArrowRight" :aria-label="`${activePreview.title}放大预览`" @click.self="closePreview">
      <div ref="lightboxPanel" class="result-lightbox__panel">
        <div class="result-lightbox__header"><div><small>{{ activePreview.english }}</small><strong>{{ activePreview.title }}</strong></div><button ref="closePreviewButton" type="button" aria-label="关闭预览" @click="closePreview">关闭 ×</button></div>
        <div class="result-lightbox__viewer">
          <button type="button" class="result-lightbox__arrow" aria-label="查看上一阶段图片" title="上一阶段 · ←" @click="movePreview(-1)">←</button>
          <img v-if="activePreview.filename" :src="fileUrl(activePreview.filename)" :alt="activePreview.description">
          <button type="button" class="result-lightbox__arrow" aria-label="查看下一阶段图片" title="下一阶段 · →" @click="movePreview(1)">→</button>
        </div>
        <div class="result-lightbox__footer"><p>{{ activePreview.description }} · 原始生成文件，未经前端修图</p><span>{{ previewPosition + 1 }} / {{ previewableStages.length }} · 可用 ← → 切换</span></div>
      </div>
    </div>
  </section>
</template>
