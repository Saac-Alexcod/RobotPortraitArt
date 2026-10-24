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
import {
  MAX_RECENT_PORTRAITS,
  clearRecentPortraits,
  listRecentPortraits,
  removeRecentPortrait,
  saveRecentPortrait,
} from './recentHistory'
import type { RecentPortrait } from './recentHistory'

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024
const POLL_INTERVAL_MS = 900
const POLL_RETRY_MS = 1500
const MAX_POLL_FAILURES = 10

// Curated synthetic portraits. Vite serves these copies from frontend/public/.
const sampleExamples = [
  { filename: 'portrait_sample.jpg', label: '短发正面', hint: '首次体验推荐', challenge: false },
  { filename: 'portrait_long_hair.jpg', label: '长发人像', hint: '观察发丝细节', challenge: false },
  { filename: 'portrait_glasses.jpg', label: '眼镜人像', hint: '观察眼镜轮廓', challenge: false },
  { filename: 'portrait_profile_challenge.jpg', label: '侧脸挑战', hint: 'GAN 检测可能失败', challenge: true },
] as const

type SampleExample = (typeof sampleExamples)[number]
function sampleImageUrl(sample: SampleExample): string {
  return `${import.meta.env.BASE_URL}examples/${sample.filename}`
}


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
const selectedSampleName = ref<string | null>(null)
const isPreviewOpen = ref(false)
const previewOpenButton = ref<HTMLButtonElement | null>(null)
const previewCloseButton = ref<HTMLButtonElement | null>(null)
const loadingSampleName = ref<string | null>(null)
let sampleRequestToken = 0
let historyRestoreToken = 0
let historyWriteQueue: Promise<void> = Promise.resolve()
const recentPortraits = ref<RecentPortrait[]>([])
const activeRecentId = ref<string | null>(null)
const historyLoading = ref(true)
const historyWarning = ref<string | null>(null)
const recentPreviewUrls = new Map<string, string>()
const submitElapsedMs = ref<number | null>(null)

function recentImageUrl(item: RecentPortrait): string {
  let url = recentPreviewUrls.get(item.id)
  if (!url) {
    url = URL.createObjectURL(item.file)
    recentPreviewUrls.set(item.id, url)
  }
  return url
}

function updateRecentList(items: RecentPortrait[]): void {
  const unique = new Map(items.map(item => [item.id, item]))
  recentPortraits.value = Array.from(unique.values())
    .sort((a, b) => b.savedAt - a.savedAt)
    .slice(0, MAX_RECENT_PORTRAITS)
  const keep = new Set(recentPortraits.value.map(item => item.id))
  for (const [id, url] of recentPreviewUrls) {
    if (!keep.has(id)) {
      URL.revokeObjectURL(url)
      recentPreviewUrls.delete(id)
    }
  }
}

function persistRecent(item: RecentPortrait): void {
  historyWriteQueue = historyWriteQueue
    .catch(() => {})
    .then(() => saveRecentPortrait(item))
    .catch(() => {
      historyWarning.value = '本地存储不可用，最近图片只会保留在当前页面。'
    })
}

async function loadRecentHistory(): Promise<void> {
  try {
    const stored = await listRecentPortraits()
    updateRecentList([...stored, ...recentPortraits.value])
  } catch {
    historyWarning.value = '本地存储不可用，最近图片只会保留在当前页面。'
  } finally {
    historyLoading.value = false
  }
}

function rememberFile(file: File, sampleName: string | null): void {
  const now = Date.now()
  // Selecting the same built-in sample returns to its existing entry.
  const existing = sampleName
    ? recentPortraits.value.find(item => item.sampleName === sampleName)
    : null
  const item: RecentPortrait = {
    id: existing?.id ?? (crypto.randomUUID?.() ?? `${now}-${Math.random()}`),
    name: file.name,
    type: file.type || 'application/octet-stream',
    file,
    savedAt: now,
    sampleName,
    jobId: existing?.jobId ?? null,
    method: existing?.method ?? null,
    vectorMode: existing?.vectorMode ?? null,
    size: existing?.size ?? null,
  }
  activeRecentId.value = item.id
  updateRecentList([item, ...recentPortraits.value.filter(entry => entry.id !== item.id)])
  persistRecent(item)
}

function linkJobToRecent(response: JobCreatedResponse, id: string | null): void {
  const previous = recentPortraits.value.find(item => item.id === id)
  if (!previous) return
  const item: RecentPortrait = {
    ...previous,
    jobId: response.job_id,
    method: response.method,
    vectorMode: response.vector_mode,
    size: response.size,
    savedAt: Date.now(),
  }
  updateRecentList([item, ...recentPortraits.value.filter(record => record.id !== id)])
  persistRecent(item)
}

async function restoreRecent(item: RecentPortrait): Promise<void> {
  if (isSubmitting.value) return
  const file = new File([item.file], item.name, { type: item.type })
  useFile(file, { remember: false, sampleName: item.sampleName })
  if (!selectedFile.value) return
  const token = historyRestoreToken
  activeRecentId.value = item.id
  selectedSampleName.value = item.sampleName
  if (!item.jobId) return
  try {
    const job = await getJob(item.jobId)
    if (token !== historyRestoreToken || activeRecentId.value !== item.id) return
    selectedMethod.value = job.method
    selectedVectorMode.value = job.vector_mode
    selectedSize.value = job.size
    // Set after the parameter watchers clear the previous job state.
    await nextTick()
    if (token !== historyRestoreToken || activeRecentId.value !== item.id) return
    createdJob.value = {
      job_id: job.job_id,
      status: job.status,
      method: job.method,
      vector_mode: job.vector_mode,
      size: job.size,
    }
    startPolling(job.job_id)
  } catch {
    if (token === historyRestoreToken && activeRecentId.value === item.id) {
      historyWarning.value = '图片已恢复，但对应任务可能已从后端输出目录删除。'
    }
  }
}

async function removeHistoryItem(id: string): Promise<void> {
  updateRecentList(recentPortraits.value.filter(item => item.id !== id))
  if (activeRecentId.value === id) activeRecentId.value = null
  try {
    await historyWriteQueue.catch(() => {})
    await removeRecentPortrait(id)
  }
  catch { historyWarning.value = '删除本地历史失败，请检查浏览器存储权限。' }
}

async function clearAllHistory(): Promise<void> {
  if (!window.confirm('清除本浏览器保存的最近图片与任务入口？不会删除后端生成文件。')) return
  updateRecentList([])
  activeRecentId.value = null
  try {
    await historyWriteQueue.catch(() => {})
    await clearRecentPortraits()
  }
  catch { historyWarning.value = '清除本地历史失败。' }
}


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
    && loadingSampleName.value === null
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

async function openFullPreview(): Promise<void> {
  if (!previewUrl.value) return
  isPreviewOpen.value = true
  await nextTick()
  previewCloseButton.value?.focus()
}

function closeFullPreview(): void {
  if (!isPreviewOpen.value) return
  isPreviewOpen.value = false
  void nextTick(() => previewOpenButton.value?.focus())
}

function handlePreviewKeydown(event: KeyboardEvent): void {
  if (!isPreviewOpen.value) return
  if (event.key === 'Escape') {
    event.preventDefault()
    closeFullPreview()
  } else if (event.key === 'Tab') {
    // Keep keyboard focus inside the one-control preview dialog.
    event.preventDefault()
    previewCloseButton.value?.focus()
  }
}

function revokePreview(): void {
  isPreviewOpen.value = false
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
}

function clearSelection(): void {
  if (isSubmitting.value) return
  historyRestoreToken += 1
  activeRecentId.value = null
  sampleRequestToken += 1
  loadingSampleName.value = null
  selectedSampleName.value = null
  resetJobTracking()
  revokePreview()
  selectedFile.value = null
  fileError.value = null
  submitError.value = null

  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

function useFile(file: File, options: { remember?: boolean; sampleName?: string | null } = {}): void {
  if (isSubmitting.value) return
  historyRestoreToken += 1
  activeRecentId.value = null
  // Any manual upload or new file selection invalidates an older sample fetch.
  sampleRequestToken += 1
  loadingSampleName.value = null
  selectedSampleName.value = null
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
  if (options.remember !== false) rememberFile(file, options.sampleName ?? null)
}

async function chooseSample(sample: SampleExample): Promise<void> {
  if (isSubmitting.value) return
  const token = ++sampleRequestToken
  loadingSampleName.value = sample.filename
  fileError.value = null

  try {
    const response = await fetch(sampleImageUrl(sample))
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    const blob = await response.blob()
    if (token !== sampleRequestToken) return
    // A missing static asset may return Vite's index.html with HTTP 200.
    // Validate JPEG bytes rather than trusting the response status or MIME.
    const signature = new Uint8Array(await blob.slice(0, 3).arrayBuffer())
    if (
      signature.length !== 3
      || signature[0] !== 0xff
      || signature[1] !== 0xd8
      || signature[2] !== 0xff
    ) {
      throw new Error('示例资源不是有效 JPEG（请确认 frontend/public/examples 路径）')
    }
    if (token !== sampleRequestToken) return

    // Send exactly the same multipart File object as a manual user upload.
    const file = new File([blob], sample.filename, { type: 'image/jpeg' })
    useFile(file, { sampleName: sample.filename })
    if (selectedFile.value) selectedSampleName.value = sample.filename
    if (fileInput.value) {
      fileInput.value.value = ''
    }
  } catch (error: unknown) {
    if (token === sampleRequestToken) {
      fileError.value = `示例图片读取失败，请重试或从本地选择：${error instanceof Error ? error.message : '未知错误'}`
    }
  } finally {
    if (token === sampleRequestToken) {
      loadingSampleName.value = null
    }
  }
}

function onFileChange(event: Event): void {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]

  if (file && !isSubmitting.value) {
    useFile(file)
  }
  // Allow choosing the same file again; the input is not the source of truth.
  input.value = ''
}

function onDrop(event: DragEvent): void {
  const file = event.dataTransfer?.files?.[0]

  if (file && !isSubmitting.value) {
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
  submitElapsedMs.value = null
  const submittedAt = performance.now()
  const submittedHistoryId = activeRecentId.value

  try {
    const response = await createJob({
      image: selectedFile.value,
      method: selectedMethod.value,
      vectorMode: selectedVectorMode.value,
      size: selectedSize.value,
    })

    submitElapsedMs.value = Math.round(performance.now() - submittedAt)
    linkJobToRecent(response, submittedHistoryId)
    createdJob.value = response
    startPolling(response.job_id)
  } catch (error: unknown) {
    submitElapsedMs.value = Math.round(performance.now() - submittedAt)
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

async function loadBackendStatus(refresh = false): Promise<void> {
  isCheckingBackend.value = true
  backendError.value = null

  try {
    const [
      healthResponse,
      methodsResponse,
    ] = await Promise.all([
      getHealth(),
      getMethods(refresh),
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
  if (isSubmitting.value) return '正在检查运行环境并提交图片…'
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
  void loadRecentHistory()
  window.addEventListener('keydown', handlePreviewKeydown)
  void loadBackendStatus()
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handlePreviewKeydown)
  stopPolling()
  revokePreview()
  for (const url of recentPreviewUrls.values()) URL.revokeObjectURL(url)
  recentPreviewUrls.clear()
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
            <input ref="fileInput" :disabled="isSubmitting" class="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" @change="onFileChange">
            <button class="drop-zone studio-drop" type="button" :disabled="isSubmitting" @click="chooseFile" @dragover.prevent @drop.prevent="onDrop" aria-label="点击或拖放上传人像照片">
              <img v-if="previewUrl" class="studio-drop__image" :src="previewUrl" alt="当前已选择的人像照片（完整画幅）">
              <span v-else class="drop-zone__empty studio-drop__empty"><span class="studio-drop__icon" aria-hidden="true">＋</span><strong>将照片拖到这里</strong><small>或者点击浏览本地文件</small></span>
              <span v-if="previewUrl" class="studio-drop__replace">更换照片 ↗</span>
            </button>
            <div v-if="selectedFile" class="file-meta">
              <div class="file-meta__info"><strong>{{ selectedFile.name }}</strong><span>{{ formatBytes(selectedFile.size) }}</span></div>
              <div class="file-meta__actions">
                <button ref="previewOpenButton" type="button" class="text-button" @click="openFullPreview">查看整图</button>
                <button type="button" class="text-button" @click="clearSelection">移除</button>
              </div>
            </div>
            <div class="studio-sample-gallery" aria-label="可直接使用的示例人像">
              <div class="studio-sample-gallery__heading">
                <strong>试用示例人像</strong>
                <span>点击照片，即可加入上方创作区</span>
              </div>
              <div class="studio-sample-gallery__grid">
                <button
                  v-for="sample in sampleExamples"
                  :key="sample.filename"
                  type="button"
                  class="studio-sample-card"
                  :disabled="isSubmitting"
                  :class="{ 'studio-sample-card--selected': selectedSampleName === sample.filename }"
                  :aria-pressed="selectedSampleName === sample.filename"
                  :aria-label="`使用${sample.label}${sample.challenge ? '（挑战样例）' : ''}`"
                  @click="chooseSample(sample)"
                >
                  <img :src="sampleImageUrl(sample)" :alt="sample.label" loading="lazy" width="68" height="68">
                  <span class="studio-sample-card__text">
                    <strong>{{ sample.label }}</strong>
                    <small>{{ loadingSampleName === sample.filename ? '正在载入…' : sample.hint }}</small>
                  </span>
                  <span v-if="selectedSampleName === sample.filename" class="studio-sample-card__check" aria-hidden="true">✓</span>
                </button>
              </div>
              <p class="studio-sample-gallery__note">均为合成人像；侧脸用于挑战测试，APDrawingGAN 可能无法检测。</p>
            </div>
            <section class="studio-recent" aria-label="最近图片与任务">
              <div class="studio-recent__heading">
                <strong>最近使用</strong>
                <button v-if="recentPortraits.length" type="button" class="text-button" @click="clearAllHistory">清空本地记录</button>
              </div>
              <p class="studio-recent__note">最多保留 6 张图片及其最近一次任务。仅保存在本浏览器，可随时删除；后端生成文件不受影响。</p>
              <p v-if="historyLoading" class="studio-recent__note">正在读取本地历史…</p>
              <div v-if="recentPortraits.length" class="studio-recent__grid">
                <div v-for="item in recentPortraits" :key="item.id" class="studio-recent__item" :class="{ 'studio-recent__item--active': activeRecentId === item.id }">
                  <button type="button" :disabled="isSubmitting" class="studio-recent__restore" :aria-label="`恢复${item.name}${item.jobId ? '及其任务' : ''}`" @click="restoreRecent(item)">
                    <img :src="recentImageUrl(item)" :alt="item.name" loading="lazy">
                    <span><strong :title="item.name">{{ item.name }}</strong><small>{{ item.jobId ? '可恢复任务结果' : '已保存的图片' }}</small></span>
                  </button>
                  <button type="button" class="studio-recent__remove" :aria-label="`删除${item.name}的本地历史`" @click="removeHistoryItem(item.id)">×</button>
                </div>
              </div>
              <p v-if="historyWarning" class="studio-recent__warning" role="status">{{ historyWarning }}</p>
            </section>
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
            <div v-if="!backendOnline && !isCheckingBackend" class="studio-setup-alert" role="alert"><span>后端当前不可用</span><button type="button" @click="loadBackendStatus(true)">重试连接 ↗</button></div>
            <article v-if="createdJob" class="job-created studio-job-status" aria-live="polite">
              <div class="studio-job-status__top"><strong>{{ jobStatus?.status === 'completed' ? '生成完成' : jobStatus?.status === 'failed' ? '生成失败' : '正在处理' }}</strong><span>{{ createdJob.method }} · {{ createdJob.vector_mode }} · {{ createdJob.size }} px</span></div>
              <p>{{ currentStageText }}</p>
              <div v-if="jobStatus?.status !== 'completed' && jobStatus?.status !== 'failed'" class="studio-job-pulse" aria-hidden="true"><span></span></div>
              <button v-if="jobStatus?.status === 'completed'" type="button" class="studio-job-to-result" @click="jumpTo('generation-results')"><span>查看本次作品</span><span aria-hidden="true">→</span></button>
              <p v-if="submitElapsedMs !== null" class="studio-submit-timing">服务器接收任务耗时 {{ (submitElapsedMs / 1000).toFixed(2) }} s（不含实际生成）</p>
              <p v-if="pollError" class="job-live-error">{{ pollError }} <button v-if="jobStatus?.status !== 'completed' && jobStatus?.status !== 'failed'" type="button" class="text-button" @click="startPolling(createdJob.job_id)">重新连接</button></p>
              <p v-if="jobStatus?.error" class="job-live-error">{{ jobStatus.error.type }}：{{ jobStatus.error.message }}</p>
              <details class="studio-job-metadata"><summary>任务信息</summary><code>{{ createdJob.job_id }}</code><p>状态：{{ jobStatus?.status ?? createdJob.status }}</p></details>
            </article>
          </form>
        </div>
      </section>

      <JobResults v-if="jobStatus?.status === 'completed'" :job="jobStatus" :manifest="manifest" :manifest-error="pollError" />

      <div v-if="isPreviewOpen && previewUrl" class="studio-photo-preview" role="dialog" aria-modal="true" aria-label="上传图片完整预览" @click.self="closeFullPreview">
        <div class="studio-photo-preview__panel">
          <div class="studio-photo-preview__header"><strong>原始照片 · 完整画幅</strong><button ref="previewCloseButton" type="button" @click="closeFullPreview" aria-label="关闭完整图片预览">关闭 ×</button></div>
          <img :src="previewUrl" :alt="selectedFile ? `完整预览：${selectedFile.name}` : '完整预览图片'">
          <p>仅用于查看当前选择的照片，不会裁剪、压缩或修改上传文件。</p>
        </div>
      </div>

      <section id="system-info" class="studio-system">
        <details class="studio-system__details">
          <summary><span><span class="studio-eyebrow">SYSTEM</span><strong>系统状态与技术信息</strong><small>运行状态、生成器与接口诊断</small></span><span class="studio-system__arrow">⌄</span></summary>
          <div class="studio-system__content"><div class="studio-system__toolbar"><span>{{ backendLabel }}</span><button class="button button--secondary" type="button" :disabled="isCheckingBackend" @click="loadBackendStatus(true)">{{ isCheckingBackend ? '正在检查…' : '重新检查' }}</button></div>
            <div class="live-grid"><article class="api-card"><div class="api-card__header"><code>GET /api/health</code><span class="api-status" :class="{ 'api-status--ok': backendOnline, 'api-status--error': !isCheckingBackend && !backendOnline }">{{ isCheckingBackend ? 'Checking' : backendOnline ? 'Online' : 'Offline' }}</span></div><dl class="api-kv"><div><dt>status</dt><dd>{{ health?.status ?? '—' }}</dd></div><div><dt>service</dt><dd>{{ health?.service ?? '—' }}</dd></div></dl><p v-if="backendError" class="api-error">{{ backendError }}</p></article>
            <article class="api-card"><div class="api-card__header"><code>GET /api/methods</code><span class="api-status">{{ availableMethods.length }}/{{ methods.length }} available</span></div><div v-if="isCheckingBackend" class="api-empty">正在读取生成器状态…</div><div v-else-if="backendError" class="api-empty">等待后端连接。</div><div v-else class="method-list"><div v-for="method in methods" :key="method.name" class="method-row"><div><strong>{{ methodDisplayName(method.name) }}</strong><p>{{ backendDetail(method) }}</p></div><span class="method-badge" :class="{ 'method-badge--available': method.available, 'method-badge--unavailable': !method.available }">{{ method.available ? '可用' : '不可用' }}</span></div></div></article></div>
          </div>
        </details>
      </section>
    </main>
    <footer class="studio-footer"><strong>RobotPortraitArt</strong><span>Vue 3 · FastAPI · APDrawingGAN · Centerline · SVG</span></footer>
  </div>
</template>
