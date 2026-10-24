<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  ref,
  watch,
} from 'vue'

import { getJobFileUrl } from '../api/client'

type AnimationPhase =
  | 'idle'
  | 'penup'
  | 'drawing'
  | 'complete'

type StrokeLinecap =
  | 'butt'
  | 'round'
  | 'square'
  | 'inherit'

type StrokeLinejoin =
  | 'miter'
  | 'round'
  | 'bevel'
  | 'inherit'

interface SvgPathRecord {
  d: string
  strokeWidth: number
  linecap: StrokeLinecap
  linejoin: StrokeLinejoin
}

interface Point {
  x: number
  y: number
}

interface PathGeometry {
  length: number
  start: Point
  end: Point
}

const props = defineProps<{
  jobId: string
  filename: string
}>()

const DRAW_SPEED_PX_PER_SECOND = 680
const PEN_UP_SPEED_PX_PER_SECOND = 1800

const svgRoot = ref<SVGSVGElement | null>(null)
const viewBox = ref('0 0 512 512')
const paths = ref<SvgPathRecord[]>([])
const geometry = ref<PathGeometry[]>([])
const loadState = ref<'loading' | 'ready' | 'error'>('loading')
const loadError = ref<string | null>(null)

const currentIndex = ref(0)
const phase = ref<AnimationPhase>('idle')
const phaseProgress = ref(0)
const isPlaying = ref(false)
const speedMultiplier = ref(2)

let animationFrame: number | null = null
let lastTimestamp: number | null = null
let pathElements: SVGPathElement[] = []

const speedOptions = [0.5, 1, 2, 4]

const totalPaths = computed(
  () => paths.value.length,
)

const completedPaths = computed(() => {
  if (phase.value === 'complete') {
    return totalPaths.value
  }

  if (phase.value === 'penup') {
    return currentIndex.value
  }

  if (phase.value === 'drawing') {
    return currentIndex.value
  }

  return 0
})

const overallProgress = computed(() => {
  if (totalPaths.value === 0) {
    return 0
  }

  if (phase.value === 'complete') {
    return 100
  }

  let completedEquivalent = completedPaths.value

  if (phase.value === 'drawing') {
    completedEquivalent += phaseProgress.value
  }

  return Math.min(
    100,
    (completedEquivalent / totalPaths.value) * 100,
  )
})

const statusLabel = computed(() => {
  if (loadState.value === 'loading') {
    return '正在读取 SVG'
  }

  if (loadState.value === 'error') {
    return 'SVG 不可用'
  }

  if (phase.value === 'complete') {
    return '绘制完成'
  }

  if (phase.value === 'penup') {
    return (
      `抬笔移动 ${currentIndex.value}`
      + ` → ${currentIndex.value + 1}`
    )
  }

  if (phase.value === 'drawing') {
    return (
      `绘制第 ${currentIndex.value + 1} 笔`
      + ` / ${totalPaths.value}`
    )
  }

  return `${totalPaths.value} 笔待绘制`
})

const playButtonLabel = computed(() => {
  if (phase.value === 'complete') {
    return '重新播放'
  }

  return isPlaying.value
    ? '暂停'
    : '开始绘制'
})

const currentPoint = computed<Point | null>(() => {
  if (
    geometry.value.length === 0
    || pathElements.length === 0
  ) {
    return null
  }

  if (phase.value === 'idle') {
    return geometry.value[0]?.start ?? null
  }

  if (phase.value === 'complete') {
    return (
      geometry.value[geometry.value.length - 1]?.end
      ?? null
    )
  }

  const item = geometry.value[currentIndex.value]

  if (!item) {
    return null
  }

  if (phase.value === 'penup') {
    const previous = geometry.value[currentIndex.value - 1]

    if (!previous) {
      return item.start
    }

    return {
      x: previous.end.x
        + (item.start.x - previous.end.x)
        * phaseProgress.value,
      y: previous.end.y
        + (item.start.y - previous.end.y)
        * phaseProgress.value,
    }
  }

  const element = pathElements[currentIndex.value]

  if (!element) {
    return item.start
  }

  const distance = (
    item.length * phaseProgress.value
  )

  const point = element.getPointAtLength(distance)

  return {
    x: point.x,
    y: point.y,
  }
})

const penUpLine = computed(() => {
  if (
    phase.value !== 'penup'
    || currentIndex.value <= 0
  ) {
    return null
  }

  const previous = geometry.value[currentIndex.value - 1]
  const next = geometry.value[currentIndex.value]

  if (!previous || !next) {
    return null
  }

  return {
    x1: previous.end.x,
    y1: previous.end.y,
    x2: previous.end.x
      + (next.start.x - previous.end.x)
      * phaseProgress.value,
    y2: previous.end.y
      + (next.start.y - previous.end.y)
      * phaseProgress.value,
  }
})

function parseStrokeLinecap(
  value: string | null,
): StrokeLinecap {
  if (
    value === 'butt'
    || value === 'round'
    || value === 'square'
    || value === 'inherit'
  ) {
    return value
  }

  return 'round'
}

function parseStrokeLinejoin(
  value: string | null,
): StrokeLinejoin {
  if (
    value === 'miter'
    || value === 'round'
    || value === 'bevel'
    || value === 'inherit'
  ) {
    return value
  }

  return 'round'
}

function stopFrame(): void {
  if (animationFrame !== null) {
    window.cancelAnimationFrame(animationFrame)
    animationFrame = null
  }

  lastTimestamp = null
}

function pause(): void {
  isPlaying.value = false
  stopFrame()
}

function resetSimulation(): void {
  pause()
  currentIndex.value = 0
  phase.value = 'idle'
  phaseProgress.value = 0
}

function pathDashOffset(index: number): number {
  if (phase.value === 'complete') {
    return 0
  }

  if (index < currentIndex.value) {
    return 0
  }

  if (index > currentIndex.value) {
    return 1
  }

  if (phase.value === 'drawing') {
    return 1 - phaseProgress.value
  }

  return 1
}

function penUpDistance(index: number): number {
  if (index <= 0) {
    return 0
  }

  const previous = geometry.value[index - 1]
  const current = geometry.value[index]

  if (!previous || !current) {
    return 0
  }

  return Math.hypot(
    current.start.x - previous.end.x,
    current.start.y - previous.end.y,
  )
}

function currentPhaseDurationMs(): number {
  const speed = Math.max(
    0.1,
    speedMultiplier.value,
  )

  if (phase.value === 'drawing') {
    const item = geometry.value[currentIndex.value]
    const length = item?.length ?? 0

    return Math.max(
      24,
      (
        length
        / (DRAW_SPEED_PX_PER_SECOND * speed)
      ) * 1000,
    )
  }

  if (phase.value === 'penup') {
    return Math.max(
      20,
      (
        penUpDistance(currentIndex.value)
        / (PEN_UP_SPEED_PX_PER_SECOND * speed)
      ) * 1000,
    )
  }

  return 1
}

function advancePhase(): void {
  if (phase.value === 'drawing') {
    if (
      currentIndex.value
      >= totalPaths.value - 1
    ) {
      phase.value = 'complete'
      phaseProgress.value = 1
      isPlaying.value = false
      stopFrame()
      return
    }

    currentIndex.value += 1
    phase.value = 'penup'
    phaseProgress.value = 0
    return
  }

  if (phase.value === 'penup') {
    phase.value = 'drawing'
    phaseProgress.value = 0
  }
}

function animationTick(timestamp: number): void {
  if (!isPlaying.value) {
    return
  }

  if (lastTimestamp === null) {
    lastTimestamp = timestamp
    animationFrame = window.requestAnimationFrame(
      animationTick,
    )
    return
  }

  const elapsed = timestamp - lastTimestamp
  lastTimestamp = timestamp

  const duration = currentPhaseDurationMs()

  phaseProgress.value = Math.min(
    1,
    phaseProgress.value + elapsed / duration,
  )

  if (phaseProgress.value >= 1) {
    advancePhase()
  }

  if (isPlaying.value) {
    animationFrame = window.requestAnimationFrame(
      animationTick,
    )
  }
}

function play(): void {
  if (
    loadState.value !== 'ready'
    || totalPaths.value === 0
  ) {
    return
  }

  if (phase.value === 'complete') {
    resetSimulation()
  }

  if (phase.value === 'idle') {
    phase.value = 'drawing'
    currentIndex.value = 0
    phaseProgress.value = 0
  }

  if (isPlaying.value) {
    return
  }

  isPlaying.value = true
  stopFrame()
  animationFrame = window.requestAnimationFrame(
    animationTick,
  )
}

function togglePlayback(): void {
  if (isPlaying.value) {
    pause()
    return
  }

  play()
}

function restart(): void {
  resetSimulation()
  play()
}

async function loadSvg(): Promise<void> {
  pause()
  loadState.value = 'loading'
  loadError.value = null
  paths.value = []
  geometry.value = []
  pathElements = []

  try {
    const response = await fetch(
      getJobFileUrl(
        props.jobId,
        props.filename,
      ),
      {
        headers: {
          Accept: 'image/svg+xml,text/plain',
        },
      },
    )

    if (!response.ok) {
      throw new Error(
        `SVG request returned HTTP ${response.status}`,
      )
    }

    const svgText = await response.text()
    const documentNode = new DOMParser().parseFromString(
      svgText,
      'image/svg+xml',
    )

    if (documentNode.querySelector('parsererror')) {
      throw new Error('Generated SVG could not be parsed.')
    }

    const root = documentNode.documentElement

    if (root.tagName.toLowerCase() !== 'svg') {
      throw new Error('Generated file is not an SVG document.')
    }

    const parsedViewBox = root.getAttribute('viewBox')

    if (parsedViewBox) {
      viewBox.value = parsedViewBox
    } else {
      const width = Number.parseFloat(
        root.getAttribute('width') || '512',
      )
      const height = Number.parseFloat(
        root.getAttribute('height') || '512',
      )

      viewBox.value = (
        `0 0 ${Number.isFinite(width) ? width : 512}`
        + ` ${Number.isFinite(height) ? height : 512}`
      )
    }

    paths.value = Array.from(
      root.querySelectorAll('path[d]'),
    ).map((element) => ({
      d: element.getAttribute('d') || '',
      strokeWidth: Number.parseFloat(
        element.getAttribute('stroke-width') || '1.2',
      ) || 1.2,
      linecap: parseStrokeLinecap(
        element.getAttribute('stroke-linecap'),
      ),
      linejoin: parseStrokeLinejoin(
        element.getAttribute('stroke-linejoin'),
      ),
    }))

    if (paths.value.length === 0) {
      throw new Error(
        'Generated SVG does not contain drawable paths.',
      )
    }

    // Make the parsed paths render before measuring their SVG geometry.
    loadState.value = 'ready'
    await nextTick()

    pathElements = Array.from(
      svgRoot.value?.querySelectorAll<SVGPathElement>(
        '.simulation-path',
      ) ?? [],
    )

    geometry.value = pathElements.map((element) => {
      const length = element.getTotalLength()
      const start = element.getPointAtLength(0)
      const end = element.getPointAtLength(length)

      return {
        length,
        start: {
          x: start.x,
          y: start.y,
        },
        end: {
          x: end.x,
          y: end.y,
        },
      }
    })

    if (
      pathElements.length !== paths.value.length
      || geometry.value.length !== paths.value.length
    ) {
      throw new Error(
        'SVG path geometry could not be initialized.',
      )
    }

    resetSimulation()
  } catch (error: unknown) {
    loadState.value = 'error'
    loadError.value = (
      error instanceof Error
        ? error.message
        : 'Unable to load generated SVG.'
    )
  }
}

watch(
  () => [
    props.jobId,
    props.filename,
  ],
  () => {
    void loadSvg()
  },
  {
    immediate: true,
  },
)

onBeforeUnmount(() => {
  pause()
})
</script>

<template>
  <section class="drawing-simulation">
    <div class="drawing-simulation__header">
      <div>
        <span>ROBOT DRAWING SIMULATION</span>
        <h3>真实 SVG Path 逐笔绘制</h3>
        <p>
          按后端规划后的 path 顺序播放。
          蓝色为 Pen-down，绿色虚线为 Pen-up 移动。
        </p>
      </div>

      <div class="simulation-controls">
        <button
          class="simulation-button simulation-button--primary"
          type="button"
          :disabled="loadState !== 'ready'"
          :aria-label="isPlaying ? '暂停轨迹播放' : '播放机器人绘制轨迹'"
          @click="togglePlayback"
        >
          {{ playButtonLabel }}
        </button>

        <button
          class="simulation-button"
          type="button"
          :disabled="loadState !== 'ready'"
          @click="restart"
        >
          重播
        </button>
      </div>
    </div>

    <div class="simulation-layout">
      <div class="simulation-canvas" :class="{ 'simulation-canvas--idle': !isPlaying && phase === 'idle', 'simulation-canvas--active': isPlaying }">
        <div
          v-if="loadState === 'loading'"
          class="simulation-message"
        >
          正在读取生成的 Robot SVG…
        </div>

        <div
          v-else-if="loadState === 'error'"
          class="simulation-message simulation-message--error"
        >
          {{ loadError }}
        </div>

        <svg
          v-else
          ref="svgRoot"
          class="simulation-svg"
          :viewBox="viewBox"
          preserveAspectRatio="xMidYMid meet"
          aria-label="Robot drawing animation"
        >
          <rect
            class="simulation-background"
            width="100%"
            height="100%"
          />

          <g class="simulation-ghost-layer">
            <path
              v-for="(path, index) in paths"
              :key="`ghost-${index}`"
              :d="path.d"
              fill="none"
              :stroke-width="path.strokeWidth"
              :stroke-linecap="path.linecap"
              :stroke-linejoin="path.linejoin"
            />
          </g>

          <g class="simulation-drawing-layer">
            <path
              v-for="(path, index) in paths"
              :key="`draw-${index}`"
              class="simulation-path"
              :class="{
                'simulation-path--current':
                  phase === 'drawing'
                  && index === currentIndex,
              }"
              :d="path.d"
              fill="none"
              :stroke-width="path.strokeWidth"
              :stroke-linecap="path.linecap"
              :stroke-linejoin="path.linejoin"
              pathLength="1"
              stroke-dasharray="1"
              :stroke-dashoffset="pathDashOffset(index)"
            />

            <line
              v-if="penUpLine"
              class="simulation-penup"
              :x1="penUpLine.x1"
              :y1="penUpLine.y1"
              :x2="penUpLine.x2"
              :y2="penUpLine.y2"
            />

            <g
              v-if="currentPoint"
              class="simulation-tip"
              :transform="`translate(${currentPoint.x} ${currentPoint.y})`"
            >
              <circle class="simulation-tip__halo" r="7" />
              <circle class="simulation-tip__core" r="3.4" />
            </g>
          </g>
        </svg>

        <div
          v-if="loadState === 'ready'"
          class="simulation-status-chip"
        >
          {{ statusLabel }}
        </div>
      </div>

      <aside class="simulation-panel">
        <div class="simulation-progress">
          <div class="simulation-progress__label">
            <span>绘制进度</span>
            <strong>{{ overallProgress.toFixed(1) }}%</strong>
          </div>

          <div class="simulation-progress__track">
            <div
              class="simulation-progress__fill"
              :style="{ width: `${overallProgress}%` }"
            ></div>
          </div>
        </div>

        <dl class="simulation-kv">
          <div>
            <dt>当前笔画</dt>
            <dd>
              {{
                totalPaths > 0
                  ? `${Math.min(currentIndex + 1, totalPaths)} / ${totalPaths}`
                  : '—'
              }}
            </dd>
          </div>

          <div>
            <dt>已完成</dt>
            <dd>{{ completedPaths }}</dd>
          </div>

          <div>
            <dt>当前阶段</dt>
            <dd>{{ phase === 'drawing' ? '落笔绘制' : phase === 'penup' ? '抬笔移动' : phase === 'complete' ? '已完成' : '就绪' }}</dd>
          </div>
        </dl>

        <div class="simulation-speed">
          <span>播放速度</span>

          <div class="simulation-speed__options">
            <button
              v-for="option in speedOptions"
              :key="option"
              type="button"
              :class="{
                'simulation-speed__active':
                  speedMultiplier === option,
              }"
              :aria-pressed="speedMultiplier === option"
              :aria-label="`以 ${option} 倍速度播放`"
              @click="speedMultiplier = option"
            >
              {{ option }}×
            </button>
          </div>
        </div>

        <div class="simulation-legend">
          <span>
            <i class="legend-line legend-line--draw"></i>
            Pen-down
          </span>
          <span>
            <i class="legend-line legend-line--move"></i>
            Pen-up
          </span>
          <span>
            <i class="legend-dot"></i>
            Pen tip
          </span>
        </div>

        <p class="simulation-note">
          动画直接解析当前任务生成的 SVG；
          path 顺序就是后端轨迹规划后的实际绘制顺序。
        </p>
      </aside>
    </div>
  </section>
</template>
