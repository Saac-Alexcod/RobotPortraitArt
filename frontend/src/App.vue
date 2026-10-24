<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from 'vue'

import {
  ApiError,
  createJob,
  getHealth,
  getJob,
  getManifest,
  getMethods,
} from './api/client'
import type {
  HealthResponse,
  JobCreatedResponse,
  JobStatusResponse,
  ManifestResponse,
  MethodStatus,
  VectorMode,
} from './api/types'
import JobResults from './components/JobResults.vue'

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024
const POLL_INTERVAL_MS = 900
const POLL_RETRY_MS = 1500
const MAX_POLL_FAILURES = 10


const health = ref<HealthResponse | null>(null)
const methods = ref<MethodStatus[]>([])
const isCheckingBackend = ref(true)
const backendError = ref<string | null>(null)

const selectedFile = ref<File | null>(null)
const previewUrl = ref<string | null>(null)
const selectedMethod = ref('apdrawinggan')
const selectedVectorMode = ref<VectorMode>('centerline')
const selectedSize = ref(512)
const fileError = ref<string | null>(null)
const submitError = ref<string | null>(null)
const isSubmitting = ref(false)
const createdJob = ref<JobCreatedResponse | null>(null)
const jobStatus = ref<JobStatusResponse | null>(null)
const manifest = ref<ManifestResponse | null>(null)
const pollError = ref<string | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)

let pollTimer: number | null = null
let pollFailures = 0

const backendOnline = computed(
  () => (
    !backendError.value
    && health.value?.status === 'ok'
  ),
)

const availableMethods = computed(
  () => methods.value.filter(
    (method) => method.available,
  ),
)

const selectedMethodStatus = computed(
  () => methods.value.find(
    (method) => method.name === selectedMethod.value,
  ) ?? null,
)

const backendLabel = computed(() => {
  if (isCheckingBackend.value) {
    return '正在连接 FastAPI Backend'
  }

  if (backendOnline.value) {
    return 'FastAPI Backend 在线'
  }

  return 'FastAPI Backend 未连接'
})


const canSubmit = computed(
  () => (
    backendOnline.value
    && selectedFile.value !== null
    && selectedMethodStatus.value?.available === true
    && !fileError.value
    && !isSubmitting.value
    && selectedSize.value >= 64
    && selectedSize.value <= 2048
    && (
      selectedMethod.value !== 'apdrawinggan'
      || selectedSize.value === 512
    )
  ),
)

function methodDisplayName(name: string): string {
  if (name === 'baseline') {
    return 'OpenCV Baseline'
  }

  if (name === 'apdrawinggan') {
    return 'APDrawingGAN'
  }

  return name
}

function backendDetail(method: MethodStatus): string {
  if (method.available) {
    return '后端 preflight 通过，可以用于生成任务。'
  }

  return method.reason || '当前生成器不可用。'
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`
}

function stopPolling(): void {
  if (pollTimer !== null) {
    window.clearTimeout(pollTimer)
    pollTimer = null
  }
}

function resetJobTracking(): void {
  stopPolling()
  createdJob.value = null
  jobStatus.value = null
  manifest.value = null
  pollError.value = null
  pollFailures = 0
}

async function pollJob(jobId: string): Promise<void> {
  if (createdJob.value?.job_id !== jobId) {
    return
  }

  try {
    const current = await getJob(jobId)

    if (createdJob.value?.job_id !== jobId) {
      return
    }

    jobStatus.value = current
    pollError.value = null
    pollFailures = 0

    if (current.status === 'completed') {
      try {
        manifest.value = await getManifest(jobId)
      } catch (error: unknown) {
        pollError.value = (error instanceof Error
          ? `任务已完成，但指标读取失败：${error.message}`
          : '任务已完成，但指标读取失败。')
      }
      return
    }

    if (current.status === 'failed') {
      return
    }

    pollTimer = window.setTimeout(
      () => {
        void pollJob(jobId)
      },
      POLL_INTERVAL_MS,
    )
  } catch (error: unknown) {
    pollError.value = (
      error instanceof Error
        ? error.message
        : '任务状态读取失败。'
    )

    pollFailures += 1
    if (
      createdJob.value?.job_id === jobId
      && pollFailures < MAX_POLL_FAILURES
    ) {
      pollTimer = window.setTimeout(
        () => { void pollJob(jobId) },
        Math.min(5000, POLL_RETRY_MS * pollFailures),
      )
    } else if (createdJob.value?.job_id === jobId) {
      pollError.value = '与后端的连接暂时中断，请点击「重新连接任务」。'
    }
  }
}

function startPolling(jobId: string): void {
  stopPolling()
  jobStatus.value = null
  manifest.value = null
  pollError.value = null
  pollFailures = 0
  void pollJob(jobId)
}

function revokePreview(): void {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
}

function clearSelection(): void {
  resetJobTracking()
  revokePreview()
  selectedFile.value = null
  fileError.value = null
  submitError.value = null

  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

function useFile(file: File): void {
  resetJobTracking()
  revokePreview()
  selectedFile.value = null
  fileError.value = null
  submitError.value = null

  if (!file.type.startsWith('image/')) {
    fileError.value = '请选择图片文件。'

    if (fileInput.value) {
      fileInput.value.value = ''
    }

    return
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    fileError.value = '图片超过后端默认的 10 MiB 上传限制。'

    if (fileInput.value) {
      fileInput.value.value = ''
    }

    return
  }

  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
}

function onFileChange(event: Event): void {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]

  if (file) {
    useFile(file)
  }
}

function onDrop(event: DragEvent): void {
  const file = event.dataTransfer?.files?.[0]

  if (file) {
    useFile(file)
  }
}

function chooseFile(): void {
  fileInput.value?.click()
}

function pickMethod(method: MethodStatus): void {
  if (!method.available) {
    return
  }

  selectedMethod.value = method.name
}

async function submitJob(): Promise<void> {
  if (!canSubmit.value || !selectedFile.value) {
    return
  }

  stopPolling()
  submitError.value = null
  createdJob.value = null
  jobStatus.value = null
  manifest.value = null
  pollError.value = null
  isSubmitting.value = true

  try {
    const response = await createJob({
      image: selectedFile.value,
      method: selectedMethod.value,
      vectorMode: selectedVectorMode.value,
      size: selectedSize.value,
    })

    createdJob.value = response
    startPolling(response.job_id)
  } catch (error: unknown) {
    if (error instanceof ApiError) {
      submitError.value = (
        error.code
          ? `${error.code}: ${error.message}`
          : error.message
      )
    } else if (error instanceof Error) {
      submitError.value = error.message
    } else {
      submitError.value = '创建任务失败。'
    }
  } finally {
    isSubmitting.value = false
  }
}

async function loadBackendStatus(): Promise<void> {
  isCheckingBackend.value = true
  backendError.value = null

  try {
    const [
      healthResponse,
      methodsResponse,
    ] = await Promise.all([
      getHealth(),
      getMethods(),
    ])

    health.value = healthResponse
    methods.value = methodsResponse.methods

    const current = methodsResponse.methods.find(
      (method) => method.name === selectedMethod.value,
    )

    if (!current?.available) {
      const firstAvailable = methodsResponse.methods.find(
        (method) => method.available,
      )

      if (firstAvailable) {
        selectedMethod.value = firstAvailable.name
      }
    }
  } catch (error: unknown) {
    health.value = null
    methods.value = []

    backendError.value = (
      error instanceof Error
        ? error.message
        : 'Unknown backend error'
    )
  } finally {
    isCheckingBackend.value = false
  }
}

watch(
  selectedMethod,
  (method) => {
    if (method === 'apdrawinggan') {
      selectedSize.value = 512
    }

    submitError.value = null
    resetJobTracking()
  },
)

watch(
  selectedVectorMode,
  () => {
    submitError.value = null
    resetJobTracking()
  },
)

watch(
  selectedSize,
  () => {
    submitError.value = null
    resetJobTracking()
  },
)


const currentStageText = computed(() => {
  if (isSubmitting.value) return '正在提交图片…'
  const status = jobStatus.value?.status ?? createdJob.value?.status
  if (status === 'queued') return '任务已排队，等待处理'
  if (status === 'running') return '正在生成肖像与路径…'
  if (status === 'completed') return '作品已生成，可以查看结果'
  if (status === 'failed') return '处理未完成，请查看错误信息'
  return selectedFile.value ? '图片已就绪，可以开始创作' : '选择图片，开始一次创作'
})

function jumpTo(id: string): void {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function startWithFile(): void {
  jumpTo('studio')
  chooseFile()
}

watch(
  () => jobStatus.value?.status,
  (status, previous) => {
    if (status === 'completed' && previous !== 'completed') {
      void nextTick(() => jumpTo('generation-results'))
    }
  },
)

onMounted(() => {
  void loadBackendStatus()
})

onBeforeUnmount(() => {
  stopPolling()
  revokePreview()
})
</script>

<template>
  <div class="app-shell studio-app">
    <header class="topbar studio-topbar">
      <a class="brand" href="#top" aria-label="RobotPortraitArt 回到页面顶部">
        <span class="brand__mark" aria-hidden="true">
          <svg viewBox="0 0 48 48"><path d="M13 33c6-15 17-22 25-20-3 8-9 18-20 22-5 2-9 1-11-2 2 0 4 0 6 0Z" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 35c7-1 14-5 19-11" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>
        </span>
        <span><strong>RobotPortraitArt</strong><small>AI Portrait &amp; Drawing Studio</small></span>
      </a>
      <nav class="studio-nav" aria-label="页面导航">
        <button type="button" @click="jumpTo('studio')">创作</button>
        <button type="button" :disabled="jobStatus?.status !== 'completed'" @click="jumpTo('generation-results')">作品</button>
        <button type="button" @click="jumpTo('system-info')">系统</button>
      </nav>
      <div class="studio-connection" :class="{ 'studio-connection--offline': !backendOnline, 'studio-connection--checking': isCheckingBackend }" role="status">
        <span class="status-dot"></span>{{ isCheckingBackend ? '连接中' : backendOnline ? '系统在线' : '连接异常' }}
      </div>
    </header>

    <main id="top">
      <section class="studio-intro" aria-labelledby="hero-title">
        <div class="studio-intro__copy">
          <span class="studio-eyebrow"><span class="studio-eyebrow__line"></span> AI × ROBOTIC ART</span>
          <h1 id="hero-title">从照片出发，<br><span>让线条动起来。</span></h1>
          <p>将一张人像转化为艺术线稿，再规划机器人的绘制路径。创作、比较、观察每一笔，都在同一个工作台完成。</p>
          <div class="studio-intro__actions">
            <button class="button button--primary studio-cta" type="button" @click="startWithFile">开始创作 <span aria-hidden="true">↗</span></button>
            <button class="studio-plain-link" type="button" @click="jumpTo('studio')">查看工作台 <span aria-hidden="true">↓</span></button>
          </div>
        </div>
        <div class="studio-intro__visual" aria-label="三个主要处理环节">
          <div class="studio-journey__header"><span>创作流程</span><span>从照片到机器人轨迹</span></div>
          <div class="studio-journey">
            <div class="studio-journey__item"><span class="studio-journey__index">01</span><span class="studio-journey__symbol" aria-hidden="true">◉</span><strong>人像输入</strong><small>检测与对齐</small></div>
            <div class="studio-journey__connector" aria-hidden="true">→</div>
            <div class="studio-journey__item"><span class="studio-journey__index">02</span><span class="studio-journey__symbol" aria-hidden="true">✳</span><strong>艺术生成</strong><small>APDrawingGAN</small></div>
            <div class="studio-journey__connector" aria-hidden="true">→</div>
            <div class="studio-journey__item"><span class="studio-journey__index">03</span><span class="studio-journey__symbol" aria-hidden="true">⌁</span><strong>轨迹规划</strong><small>Centerline · 2-opt</small></div>
          </div>
          <div class="studio-journey__foot"><span class="studio-journey__live"></span> 真实算法处理 · 输出文件可下载和复现</div>
        </div>
      </section>

      <section id="studio" class="studio-workspace" aria-labelledby="studio-title">
        <div class="studio-section-heading">
          <div><span class="studio-eyebrow">THE WORKSPACE</span><h2 id="studio-title">创作工作台</h2><p>上传照片，选择生成方式，剩下的交给系统。</p></div>
          <div class="studio-workspace__state" aria-live="polite"><span class="studio-workspace__state-dot"></span>{{ currentStageText }}</div>
        </div>

        <div class="job-layout studio-job-layout">
          <div class="upload-panel studio-upload">
            <div class="studio-panel-heading"><strong>01 / 选择照片</strong><span>PNG · JPG · WEBP · 10 MiB 内</span></div>
            <input ref="fileInput" class="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" @change="onFileChange">
            <button class="drop-zone studio-drop" type="button" @click="chooseFile" @dragover.prevent @drop.prevent="onDrop" aria-label="点击或拖放上传人像照片">
              <img v-if="previewUrl" :src="previewUrl" alt="当前上传人像预览">
              <span v-else class="drop-zone__empty studio-drop__empty"><span class="studio-drop__icon" aria-hidden="true">＋</span><strong>将照片拖到这里</strong><small>或者点击浏览本地文件</small></span>
              <span v-if="previewUrl" class="studio-drop__replace">更换照片 ↗</span>
            </button>
            <div v-if="selectedFile" class="file-meta"><div><strong>{{ selectedFile.name }}</strong><span>{{ formatBytes(selectedFile.size) }}</span></div><button type="button" class="text-button" @click="clearSelection">移除</button></div>
            <p v-if="fileError" class="form-error" role="alert">{{ fileError }}</p>
          </div>

          <form class="job-form studio-settings" :aria-busy="isSubmitting" @submit.prevent="submitJob">
            <div class="studio-panel-heading"><strong>02 / 创作设置</strong><span>已为主方法预选推荐配置</span></div>
            <fieldset>
              <legend>生成模型</legend>
              <div class="option-grid">
                <button v-for="method in methods" :key="method.name" class="option-card" :class="{ 'option-card--selected': selectedMethod === method.name, 'option-card--disabled': !method.available }" type="button" :disabled="!method.available" :aria-pressed="selectedMethod === method.name" @click="pickMethod(method)">
                  <span><strong>{{ methodDisplayName(method.name) }}</strong><em class="method-role">{{ method.name === 'apdrawinggan' ? '艺术生成' : '经典对照' }}</em><small>{{ method.available ? '可用于本次任务' : backendDetail(method) }}</small></span><span class="studio-option-check" aria-hidden="true">{{ selectedMethod === method.name ? '✓' : '' }}</span>
                </button>
              </div>
            </fieldset>
            <div class="studio-settings__row">
              <fieldset><legend>矢量提取</legend><div class="segmented-control"><label><input v-model="selectedVectorMode" type="radio" value="contour"><span><strong>Contour</strong><small>边界轮廓</small></span></label><label><input v-model="selectedVectorMode" type="radio" value="centerline"><span><strong>Centerline</strong><small>中心线轨迹</small></span></label></div></fieldset>
              <fieldset><legend>输出尺寸</legend><label class="size-field"><span><strong>画布大小</strong><small>{{ selectedMethod === 'apdrawinggan' ? 'GAN 固定尺寸' : '64–2048 px' }}</small></span><input v-model.number="selectedSize" type="number" min="64" max="2048" step="1" :disabled="selectedMethod === 'apdrawinggan'"></label></fieldset>
            </div>
            <button class="button button--generate studio-generate" type="submit" :disabled="!canSubmit"><span>{{ isSubmitting ? '正在提交…' : '生成肖像与绘制轨迹' }}</span><span aria-hidden="true">→</span></button>
            <p v-if="submitError" class="form-error" role="alert">{{ submitError }}</p>
            <div v-if="!backendOnline && !isCheckingBackend" class="studio-setup-alert" role="alert"><span>后端当前不可用</span><button type="button" @click="loadBackendStatus">重试连接 ↗</button></div>
            <article v-if="createdJob" class="job-created studio-job-status" aria-live="polite">
              <div class="studio-job-status__top"><strong>{{ jobStatus?.status === 'completed' ? '生成完成' : jobStatus?.status === 'failed' ? '生成失败' : '正在处理' }}</strong><span>{{ createdJob.method }} · {{ createdJob.vector_mode }} · {{ createdJob.size }} px</span></div>
              <p>{{ currentStageText }}</p>
              <div v-if="jobStatus?.status !== 'completed' && jobStatus?.status !== 'failed'" class="studio-job-pulse" aria-hidden="true"><span></span></div>
              <button v-if="jobStatus?.status === 'completed'" type="button" class="studio-job-to-result" @click="jumpTo('generation-results')"><span>查看本次作品</span><span aria-hidden="true">→</span></button>
              <p v-if="pollError" class="job-live-error">{{ pollError }} <button v-if="jobStatus?.status !== 'completed' && jobStatus?.status !== 'failed'" type="button" class="text-button" @click="startPolling(createdJob.job_id)">重新连接</button></p>
              <p v-if="jobStatus?.error" class="job-live-error">{{ jobStatus.error.type }}：{{ jobStatus.error.message }}</p>
              <details class="studio-job-metadata"><summary>任务信息</summary><code>{{ createdJob.job_id }}</code><p>状态：{{ jobStatus?.status ?? createdJob.status }}</p></details>
            </article>
          </form>
        </div>
      </section>

      <JobResults v-if="jobStatus?.status === 'completed'" :job="jobStatus" :manifest="manifest" :manifest-error="pollError" />

      <section id="system-info" class="studio-system">
        <details class="studio-system__details">
          <summary><span><span class="studio-eyebrow">SYSTEM</span><strong>系统状态与技术信息</strong><small>运行状态、生成器与接口诊断</small></span><span class="studio-system__arrow">⌄</span></summary>
          <div class="studio-system__content"><div class="studio-system__toolbar"><span>{{ backendLabel }}</span><button class="button button--secondary" type="button" :disabled="isCheckingBackend" @click="loadBackendStatus">{{ isCheckingBackend ? '正在检查…' : '重新检查' }}</button></div>
            <div class="live-grid"><article class="api-card"><div class="api-card__header"><code>GET /api/health</code><span class="api-status" :class="{ 'api-status--ok': backendOnline, 'api-status--error': !isCheckingBackend && !backendOnline }">{{ isCheckingBackend ? 'Checking' : backendOnline ? 'Online' : 'Offline' }}</span></div><dl class="api-kv"><div><dt>status</dt><dd>{{ health?.status ?? '—' }}</dd></div><div><dt>service</dt><dd>{{ health?.service ?? '—' }}</dd></div></dl><p v-if="backendError" class="api-error">{{ backendError }}</p></article>
            <article class="api-card"><div class="api-card__header"><code>GET /api/methods</code><span class="api-status">{{ availableMethods.length }}/{{ methods.length }} available</span></div><div v-if="isCheckingBackend" class="api-empty">正在读取生成器状态…</div><div v-else-if="backendError" class="api-empty">等待后端连接。</div><div v-else class="method-list"><div v-for="method in methods" :key="method.name" class="method-row"><div><strong>{{ methodDisplayName(method.name) }}</strong><p>{{ backendDetail(method) }}</p></div><span class="method-badge" :class="{ 'method-badge--available': method.available, 'method-badge--unavailable': !method.available }">{{ method.available ? '可用' : '不可用' }}</span></div></div></article></div>
          </div>
        </details>
      </section>
    </main>
    <footer class="studio-footer"><strong>RobotPortraitArt</strong><span>Vue 3 · FastAPI · APDrawingGAN · Centerline · SVG</span></footer>
  </div>
</template>
