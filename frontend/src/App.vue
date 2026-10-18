<script setup lang="ts">
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

const backendCapabilities = [
  'FastAPI job API',
  'Baseline / APDrawingGAN',
  'Contour / Centerline',
  'WSL + CUDA inference',
]
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
        Web Demo V1
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

            <div class="backend-chip">
              <span class="backend-chip__pulse"></span>
              FastAPI Backend 已完成
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
          <p>V1 先建立稳定、可扩展的 Vue3 + TypeScript UI 骨架。</p>
        </div>

        <div class="stage-grid">
          <StageCard
            v-for="stage in stages"
            :key="stage.step"
            v-bind="stage"
          />
        </div>
      </section>

      <section class="architecture-section">
        <div class="architecture-card">
          <div class="architecture-card__copy">
            <span class="architecture-card__eyebrow">V1 SCOPE</span>
            <h2>前后端边界先搭清楚</h2>
            <p>
              当前页面只负责 UI 骨架，不提前塞入业务状态管理。
              Vite 开发服务器已经预留 <code>/api</code> proxy，
              下一步 V2 直接接入 <code>/api/health</code> 和
              <code>/api/methods</code>。
            </p>
          </div>

          <div class="architecture-flow" aria-label="Frontend architecture">
            <div class="flow-node flow-node--active">
              <span>Vue3 + TS</span>
              <strong>Frontend</strong>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-node">
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
      <span>Vue3 + TypeScript Web Demo · V1</span>
    </footer>
  </div>
</template>
