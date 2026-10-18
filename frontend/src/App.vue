<script setup lang="ts">
import {
  computed,
  onMounted,
  ref,
} from 'vue'

import {
  getHealth,
  getMethods,
} from './api/client'
import type {
  HealthResponse,
  MethodStatus,
} from './api/types'
import StageCard from './components/StageCard.vue'

const stages = [
  {
    step: '01',
    title: '上传肖像',
    description: '选择人脸照片，作为艺术线稿生成和机器人绘制的输入。',
    status: 'next' as const,
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
    status: 'next' as const,
  },
]

const health = ref<HealthResponse | null>(null)
const methods = ref<MethodStatus[]>([])
const isCheckingBackend = ref(true)
const backendError = ref<string | null>(null)

const backendOnline = computed(
  () => (
    !backendError.value
    && health.value?.status === 'ok'
  ),
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

function methodDisplayName(name: string): string {
  if (name === 'baseline') {
    return 'Baseline'
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

onMounted(() => {
  void loadBackendStatus()
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
        Web Demo V2
      </div>
    </header>

    <main>
      <section class="hero">
        <div class="hero__copy">
          <div class="eyebrow">机器人肖像绘制艺术风格学习</div>

          <h1>
            从一张人脸照片，
            <span>生成机器人可执行的艺术绘制轨迹。</span>
          </h1>

          <p>
            RobotPortraitArt 将人脸预处理、GAN 肖像生成、中心线矢量化与轨迹规划
            连接成一条可交互的 Web Demo Pipeline。
          </p>

          <div class="hero__actions">
            <button class="button button--primary" type="button" disabled>
              上传图片
              <span>V3 开放</span>
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
              <span class="canvas-card__counter">431 paths</span>
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

              <div class="coordinate coordinate--x">X 312.4</div>
              <div class="coordinate coordinate--y">Y 188.7</div>
            </div>
          </div>
        </div>
      </section>

      <section class="workflow-section">
        <div class="section-heading">
          <div>
            <span>PIPELINE</span>
            <h2>Web Demo 开发路径</h2>
          </div>
          <p>
            V2 将 Vue 页面连接到真实 FastAPI，
            读取服务健康状态与 Generator 能力。
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
                {{ methods.length }} methods
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

      <section class="architecture-section">
        <div class="architecture-card">
          <div class="architecture-card__copy">
            <span class="architecture-card__eyebrow">V2 SCOPE</span>
            <h2>前后端已经开始真实联调</h2>
            <p>
              Vue 通过 Vite 的 <code>/api</code> proxy 调用 FastAPI。
              本阶段只读取 <code>/api/health</code> 与
              <code>/api/methods</code>，不提交生成任务。
              下一步 V3 再加入图片上传、参数选择与
              <code>POST /api/jobs</code>。
            </p>
          </div>

          <div class="architecture-flow" aria-label="Frontend architecture">
            <div class="flow-node flow-node--active">
              <span>Vue3 + TS</span>
              <strong>Frontend</strong>
            </div>
            <div class="flow-arrow">→</div>
            <div
              class="flow-node"
              :class="{ 'flow-node--online': backendOnline }"
            >
              <span>/api</span>
              <strong>FastAPI</strong>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-node">
              <span>Single Worker</span>
              <strong>Pipeline</strong>
            </div>
          </div>
        </div>
      </section>
    </main>

    <footer>
      <span>RobotPortraitArt</span>
      <span>Vue3 + TypeScript Web Demo · V2</span>
    </footer>
  </div>
</template>
