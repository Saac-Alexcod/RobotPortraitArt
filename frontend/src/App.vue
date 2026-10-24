<script setup lang="ts">
import {
  computed,
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
import StageCard from './components/StageCard.vue'

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024
const POLL_INTERVAL_MS = 900
const POLL_RETRY_MS = 1500

const stages = [
  {
    step: '01',
    title: '上传肖像',
    description: '选择人脸照片，作为艺术线稿生成和机器人绘制的输入。',
    status: 'ready' as const,
  },
  {
    step: '02',
    title: '生成线稿',
    description: '通过 Baseline 或 APDrawingGAN 生成肖像艺术线稿。',
    status: 'ready' as const,
  },
  {
    step: '03',
    title: '矢量化',
    description: '使用 contour 或 centerline 提取可执行的绘制路径。',
    status: 'ready' as const,
  },
  {
    step: '04',
    title: '机器人轨迹',
    description: '按规划后的 SVG path 顺序展示机器人落笔、抬笔和连续绘制。',
    status: 'ready' as const,
  },
]

const health = ref<HealthResponse | null>(null)
const methods = ref<MethodStatus[]>([])
const isCheckingBackend = ref(true)
const backendError = ref<string | null>(null)

const selectedFile = ref<File | null>(null)
const previewUrl = ref<string | null>(null)
const selectedMethod = ref('baseline')
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

const backendCapabilities = computed(() => {
  if (isCheckingBackend.value) {
    return [
      'Checking /api/health',
      'Loading /api/methods',
      'Contour / Centerline',
    ]
  }

  if (backendError.value) {
    return [
      'FastAPI unavailable',
      'Vite /api proxy ready',
      'Retry available',
    ]
  }

  return [
    ...methods.value.map(
      (method) => (
        `${methodDisplayName(method.name)} ${
          method.available ? '可用' : '不可用'
        }`
      ),
    ),
    'Contour / Centerline',
  ]
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

    if (current.status === 'completed') {
      manifest.value = await getManifest(jobId)
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

    if (createdJob.value?.job_id === jobId) {
      pollTimer = window.setTimeout(
        () => {
          void pollJob(jobId)
        },
        POLL_RETRY_MS,
      )
    }
  }
}

function startPolling(jobId: string): void {
  stopPolling()
  jobStatus.value = null
  manifest.value = null
  pollError.value = null
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

onMounted(() => {
  void loadBackendStatus()
})

onBeforeUnmount(() => {
  stopPolling()
  revokePreview()
})
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="RobotPortraitArt 首页">
        <span class="brand__mark" aria-hidden="true">
          <svg viewBox="0 0 48 48" role="img">
            <path
              d="M13 33c6-15 17-22 25-20-3 8-9 18-20 22-5 2-9 1-11-2 2 0 4 0 6 0Z"
              fill="none"
              stroke="currentColor"
              stroke-width="3"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
            <path
              d="M12 35c7-1 14-5 19-11"
              fill="none"
              stroke="currentColor"
              stroke-width="3"
              stroke-linecap="round"
            />
          </svg>
        </span>
        <span>
          <strong>RobotPortraitArt</strong>
          <small>Portrait → Art → Robot Path</small>
        </span>
      </a>

      <div class="topbar__badge">
        <span class="status-dot"></span>
        Web Demo V6
      </div>
    </header>

    <main>
      <section class="hero hero--compact">
        <div class="hero__copy">
          <div class="eyebrow">机器人肖像绘制艺术风格学习</div>

          <h1>
            上传肖像，
            <span>看机器人按照真实轨迹逐笔完成肖像。</span>
          </h1>

          <p>
            V6 在完整任务链上加入真实 SVG Path 动画与 A/B 结果对比，
            直观看到生成、矢量化与轨迹规划的变化。
          </p>

          <div class="hero__actions">
            <button
              class="button button--primary"
              type="button"
              @click="chooseFile"
            >
              选择图片
              <span>V6</span>
            </button>

            <div
              class="backend-chip"
              :class="{
                'backend-chip--error': !isCheckingBackend && !backendOnline,
                'backend-chip--loading': isCheckingBackend,
              }"
            >
              <span class="backend-chip__pulse"></span>
              {{ backendLabel }}
            </div>
          </div>

          <div class="capability-row" aria-label="Backend capabilities">
            <span
              v-for="capability in backendCapabilities"
              :key="capability"
              class="capability"
            >
              {{ capability }}
            </span>
          </div>
        </div>

        <div class="hero__visual" aria-label="肖像轨迹示意">
          <div class="canvas-card">
            <div class="canvas-card__bar">
              <div>
                <span class="canvas-card__label">CENTERLINE PREVIEW</span>
                <strong>Robot drawing path</strong>
              </div>
              <span class="canvas-card__counter">Benchmark · 431 paths</span>
            </div>

            <div class="portrait-canvas">
              <svg viewBox="0 0 420 420" role="img" aria-label="线稿肖像示意图">
                <path
                  class="portrait-path portrait-path--muted"
                  d="M138 145c6-53 39-91 81-92 55-2 93 43 92 104-1 58-35 119-91 139-52-18-86-76-82-151Z"
                />
                <path
                  class="portrait-path"
                  d="M156 150c23-23 47-26 69-9M244 141c24-13 48-9 66 9"
                />
                <path
                  class="portrait-path"
                  d="M166 164c11 9 29 10 41-1M254 163c12 10 30 8 41-1"
                />
                <path
                  class="portrait-path"
                  d="M228 171c-3 25-7 51-15 67 10 7 22 8 35 1"
                />
                <path
                  class="portrait-path"
                  d="M188 259c27 18 58 19 87 0M205 270c18 7 35 7 52 0"
                />
                <path
                  class="portrait-path portrait-path--accent"
                  d="M108 119c25-62 80-98 133-91 58 8 98 48 109 103-35-35-65-55-96-66-47-16-96 0-146 54Z"
                />
                <path
                  class="robot-arm"
                  d="M42 336h76l34-50 50 12"
                />
                <circle class="robot-joint" cx="118" cy="336" r="9" />
                <circle class="robot-joint" cx="152" cy="286" r="8" />
                <circle class="robot-joint" cx="202" cy="298" r="7" />
              </svg>
            </div>
          </div>
        </div>
      </section>

      <section class="job-section">
        <div class="section-heading">
          <div>
            <span>CREATE JOB</span>
            <h2>创建生成任务</h2>
          </div>
          <p>
            创建真实 job 后，页面会自动轮询 queued / running /
            completed / failed，并在完成后加载结果文件与 Manifest。
          </p>
        </div>

        <div class="job-layout">
          <div class="upload-panel">
            <input
              ref="fileInput"
              class="visually-hidden"
              type="file"
              accept="image/*"
              @change="onFileChange"
            >

            <button
              class="drop-zone"
              type="button"
              @click="chooseFile"
              @dragover.prevent
              @drop.prevent="onDrop"
            >
              <img
                v-if="previewUrl"
                :src="previewUrl"
                alt="上传肖像预览"
              >

              <span v-else class="drop-zone__empty">
                <strong>拖入肖像图片</strong>
                <small>或点击选择文件 · 最大 10 MiB</small>
              </span>
            </button>

            <div v-if="selectedFile" class="file-meta">
              <div>
                <strong>{{ selectedFile.name }}</strong>
                <span>{{ formatBytes(selectedFile.size) }}</span>
              </div>
              <button
                type="button"
                class="text-button"
                @click="clearSelection"
              >
                移除
              </button>
            </div>

            <p v-if="fileError" class="form-error">
              {{ fileError }}
            </p>
          </div>

          <form class="job-form" @submit.prevent="submitJob">
            <fieldset>
              <legend>Generator</legend>

              <div class="option-grid">
                <button
                  v-for="method in methods"
                  :key="method.name"
                  class="option-card"
                  :class="{
                    'option-card--selected': selectedMethod === method.name,
                    'option-card--disabled': !method.available,
                  }"
                  type="button"
                  :disabled="!method.available"
                  @click="pickMethod(method)"
                >
                  <span>
                    <strong>{{ methodDisplayName(method.name) }}</strong>
                    <small>{{ backendDetail(method) }}</small>
                  </span>

                  <span
                    class="method-badge"
                    :class="{
                      'method-badge--available': method.available,
                      'method-badge--unavailable': !method.available,
                    }"
                  >
                    {{ method.available ? 'Available' : 'Unavailable' }}
                  </span>
                </button>
              </div>
            </fieldset>

            <fieldset>
              <legend>Vector Mode</legend>

              <div class="segmented-control">
                <label>
                  <input
                    v-model="selectedVectorMode"
                    type="radio"
                    value="contour"
                  >
                  <span>
                    <strong>Contour</strong>
                    <small>边界路径</small>
                  </span>
                </label>

                <label>
                  <input
                    v-model="selectedVectorMode"
                    type="radio"
                    value="centerline"
                  >
                  <span>
                    <strong>Centerline</strong>
                    <small>机器人中心线轨迹</small>
                  </span>
                </label>
              </div>
            </fieldset>

            <fieldset>
              <legend>Output Size</legend>

              <label class="size-field">
                <span>
                  <strong>Canvas size</strong>
                  <small>
                    64–2048 px；APDrawingGAN 固定为 512 px
                  </small>
                </span>

                <input
                  v-model.number="selectedSize"
                  type="number"
                  min="64"
                  max="2048"
                  step="1"
                  :disabled="selectedMethod === 'apdrawinggan'"
                >
              </label>
            </fieldset>

            <button
              class="button button--generate"
              type="submit"
              :disabled="!canSubmit"
            >
              {{
                isSubmitting
                  ? '正在创建任务…'
                  : 'Generate Portrait'
              }}
            </button>

            <p v-if="submitError" class="form-error">
              {{ submitError }}
            </p>

            <article
              v-if="createdJob"
              class="job-created"
              aria-live="polite"
            >
              <span>LIVE JOB</span>
              <strong>{{ createdJob.job_id }}</strong>

              <dl>
                <div>
                  <dt>Status</dt>
                  <dd>{{ jobStatus?.status ?? createdJob.status }}</dd>
                </div>
                <div>
                  <dt>Method</dt>
                  <dd>{{ createdJob.method }}</dd>
                </div>
                <div>
                  <dt>Vector</dt>
                  <dd>{{ createdJob.vector_mode }}</dd>
                </div>
                <div>
                  <dt>Size</dt>
                  <dd>{{ createdJob.size }} px</dd>
                </div>
              </dl>

              <p v-if="jobStatus?.status === 'completed'">
                Worker 已完成任务，结果与 Manifest 已加载到页面下方。
              </p>
              <p v-else-if="jobStatus?.status === 'failed'">
                Worker 返回失败状态，请查看下方错误信息。
              </p>
              <p v-else>
                前端正在自动轮询后端状态，无需手动打开 Job API。
              </p>

              <p v-if="pollError" class="job-live-error">
                {{ pollError }}
              </p>

              <p
                v-if="jobStatus?.error"
                class="job-live-error"
              >
                {{ jobStatus.error.type }}:
                {{ jobStatus.error.message }}
              </p>
            </article>
          </form>
        </div>

        <JobResults
          v-if="jobStatus?.status === 'completed'"
          :job="jobStatus"
          :manifest="manifest"
        />
      </section>

      <section class="workflow-section">
        <div class="section-heading">
          <div>
            <span>PIPELINE</span>
            <h2>Web Demo 开发路径</h2>
          </div>
          <p>
            V6 已完成结果对比与真实 SVG path 机器人绘制模拟。
          </p>
        </div>

        <div class="stage-grid">
          <StageCard
            v-for="stage in stages"
            :key="stage.step"
            v-bind="stage"
          />
        </div>
      </section>

      <section class="live-status-section">
        <div class="section-heading">
          <div>
            <span>LIVE BACKEND</span>
            <h2>真实后端状态</h2>
          </div>

          <button
            class="button button--secondary"
            type="button"
            :disabled="isCheckingBackend"
            @click="loadBackendStatus"
          >
            {{ isCheckingBackend ? '检查中…' : '重新检查' }}
          </button>
        </div>

        <div class="live-grid">
          <article class="api-card">
            <div class="api-card__header">
              <code>GET /api/health</code>
              <span
                class="api-status"
                :class="{
                  'api-status--ok': backendOnline,
                  'api-status--error': !isCheckingBackend && !backendOnline,
                }"
              >
                {{
                  isCheckingBackend
                    ? 'Checking'
                    : backendOnline
                      ? 'Online'
                      : 'Offline'
                }}
              </span>
            </div>

            <dl class="api-kv">
              <div>
                <dt>status</dt>
                <dd>{{ health?.status ?? '—' }}</dd>
              </div>
              <div>
                <dt>service</dt>
                <dd>{{ health?.service ?? '—' }}</dd>
              </div>
            </dl>

            <p v-if="backendError" class="api-error">
              {{ backendError }}
            </p>
          </article>

          <article class="api-card">
            <div class="api-card__header">
              <code>GET /api/methods</code>
              <span class="api-status">
                {{ availableMethods.length }}/{{ methods.length }} available
              </span>
            </div>

            <div v-if="isCheckingBackend" class="api-empty">
              正在读取 GeneratorStatus…
            </div>

            <div
              v-else-if="backendError"
              class="api-empty"
            >
              后端连接成功后将在这里显示真实 Generator 状态。
            </div>

            <div v-else class="method-list">
              <div
                v-for="method in methods"
                :key="method.name"
                class="method-row"
              >
                <div>
                  <strong>{{ methodDisplayName(method.name) }}</strong>
                  <p>{{ backendDetail(method) }}</p>
                </div>

                <span
                  class="method-badge"
                  :class="{
                    'method-badge--available': method.available,
                    'method-badge--unavailable': !method.available,
                  }"
                >
                  {{ method.available ? 'Available' : 'Unavailable' }}
                </span>
              </div>
            </div>
          </article>
        </div>
      </section>
    </main>

    <footer>
      <span>RobotPortraitArt</span>
      <span>Vue3 + TypeScript Web Demo · V6</span>
    </footer>
  </div>
</template>
