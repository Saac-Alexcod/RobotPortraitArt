# 组员首次安装与启动（Windows，一页操作指南）

> **请从这里开始，不要先手动安装一堆 Python/WSL 包。** 自动安装器已通过 PR #15 合并到默认 `main` 分支。组员直接克隆仓库即可获取 `install.ps1`。新电脑首次部署仍需实际验收，安装器无法保证所有硬件和网络环境均成功。

## 0. 先确认电脑的前提条件

- **Windows 10/11（64 位）**，能够运行 PowerShell，已安装 Git；最好有 Windows Package Manager（`winget`）。安装程序会尝试补齐 Python、项目依赖、Node、Ubuntu 环境与模型，但**不会自动安装 Git**。
- 如果要使用 **APDrawingGAN**：需要本机有支持 WSL2 CUDA 的 **NVIDIA GPU + 正常的 Windows 驱动**，并允许启用 WSL/虚拟化。安装器不能替你更换 GPU、修改 BIOS、安装显卡驱动或绕过管理员限制。
- 确保项目所在盘和 **Windows 系统盘（通常是 C:）** 留有足够空间，网络能访问 Python、npm、PyTorch 和官方模型下载源；首次下载可能占用数 GB。

不知道电脑是否支持？可先运行 `git --version`、`winget --version`、`wsl --list --verbose`、`nvidia-smi` 查看当前状态；**某条不存在不代表必须立刻手工安装所有东西**，先看下面的安装器提示。

## 1. 克隆正确分支并运行一条命令

打开 **Windows PowerShell**（不是 Ubuntu 终端），复制整段运行：

```powershell
git clone https://github.com/Saac-Alexcod/RobotPortraitArt.git
cd RobotPortraitArt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

这是一台电脑的**首次安装入口**。安装器按顺序检测并尽可能复用/补装 Windows Python、后端依赖、Node/npm、WSL2、Linux 推理 Python/PyTorch、模型；最终**真的生成 APDrawingGAN 肖像和 SVG**，然后尝试启动前后端。

- 出现 Windows 管理员授权或 Ubuntu 首次用户初始化：按系统要求完成；**如要求重启 Windows，重启后回到项目目录，再运行同一条安装命令**。脚本会复用可验证的现有内容。
- 看到终端窗口分别显示 `Uvicorn running on http://127.0.0.1:8000` 与 `VITE ... http://127.0.0.1:5173` 时，保留这两个服务窗口；在浏览器里**手动打开** `http://127.0.0.1:5173`。
- 第一次体验推荐内置的 `portrait_sample.jpg`；选择 **APDrawingGAN + Centerline**，生成并查看图片、SVG、笔画模拟及下载。**仅看到网页能打开，还不等于 GAN 实际可用。**

## 2. 判断这次是否真正成功

成功至少包括：

1. 安装器输出 `[PASS] First-run acceptance completed.`，且未出现 `[FAILED]`。
2. 生成文件目录 `outputs/first_run_validation/portrait_sample/`（或其他通过的内置正面样例）包含肖像 PNG、SVG 和 `manifest.json`。
3. 浏览器能打开 `http://127.0.0.1:5173`，且后端 `http://127.0.0.1:8000/api/health` 正常响应。
4. 页面选择 APDrawingGAN 后，能提交并完成一次真实生成任务。

> `-NoStart` 只安装并执行真实推理验收，**不会启动网页**；`-BaselineOnly` 不需要 WSL/CUDA，但**不是 GAN 部署成功**。

## 3. 卡住或失败时，不要先卸载原来的环境

**先看失败阶段与错误码**，然后从项目根目录查看日志：

```powershell
Get-Content .\outputs\first_run_setup.log -Tail 40
Get-ChildItem .\outputs\installer_logs -File | Sort-Object LastWriteTime -Descending | Select-Object -First 8 Name,LastWriteTime
```

| 现象 | 先怎么处理 |
|---|---|
| 找不到 `git` | 先安装 Git for Windows；安装器以代码已克隆为前提。 |
| `wsl.exe unavailable` | 当前安装器**不会自动启用缺失的 WSL Windows 组件**。按微软官方 WSL 安装说明启用，必要时重启，然后重跑原命令。 |
| Windows 要求 UAC、重启或初始化 Ubuntu | 依提示完成，回到仓库目录后原命令再跑一次。 |
| `RPA-E210 / E230 / E250` | 分别是 Windows Python、Node/npm、WSL PyTorch/CUDA；打开 `installer_logs` 对应步骤日志。 |
| 下载超时 / HTTP 错误 / 空间不足 | 检查网络、代理和 C: / 项目盘剩余空间；不要把 `.part` 下载残片当模型用。 |
| WSL 报 `localhost` 代理警告 | 只是警告且最终测试通过时不必处理；下载失败时才查 WSL 与 Windows 的代理连通性。 |

若你**已经有可正常运行的 GAN Python** 且安装器未自动找到它，可指定当前用户的 WSL 路径（不要复制其他成员的 `/home/用户名`）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -APDPython '/home/YOUR_USER/.venvs/robotportrait-apd/bin/python'
```

如使用可信的本机 HTTP 代理，可使用 `-Proxy 'http://127.0.0.1:7890'`，但需要换成实际端口；WSL NAT 下 Windows localhost 代理**不保证**可直接使用。不要把带账号密码的代理 URL 提交到 Git 或发送到群里。

问题没解决时，记录 `Stage`、`RPA-...` 错误码，发送**清理敏感信息后的**日志末尾给项目维护者。完整步骤请看 [故障排查手册](INSTALL_TROUBLESHOOTING.md)。

## 4. 其他模式和更多说明

```powershell
# 只读检查 WSL 候选（已有 Windows Python 时可使用）
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -CheckWSLOnly

# 环境 + 模型 + 真实 GAN 推理，但不启动网页
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -NoStart

# 无 NVIDIA GPU 时仅试用 OpenCV Baseline（不表示 GAN 成功）
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -BaselineOnly
```

- [自动安装过程和限制](AUTOMATIC_INSTALL.md)
- [完整首次安装说明](FIRST_RUN_SETUP.md)
- [详细故障与日志处理](INSTALL_TROUBLESHOOTING.md)
- [模型和第三方资源](MODEL_SETUP.md)

**使用范围**：该程序输出二维 SVG 模拟绘制路径，并非实体机器人的硬件控制程序。程序默认绑定到本机 `127.0.0.1`，请不要直接暴露到公网。
