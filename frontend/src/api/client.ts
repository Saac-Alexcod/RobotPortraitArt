import type {
  CreateJobRequest,
  HealthResponse,
  JobCreatedResponse,
  JobStatusResponse,
  ManifestResponse,
  MethodsResponse,
} from './types'

export class ApiError extends Error {
  status: number
  code: string | null

  constructor(
    message: string,
    status: number,
    code: string | null = null,
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

interface ApiErrorPayload {
  detail?: {
    code?: string
    message?: string
  } | string | Array<{
    msg?: string
  }>
}

async function readApiError(
  response: Response,
): Promise<ApiError> {
  let payload: ApiErrorPayload | null = null

  try {
    payload = await response.json() as ApiErrorPayload
  } catch {
    return new ApiError(
      `HTTP ${response.status}`,
      response.status,
    )
  }

  const detail = payload?.detail

  if (
    detail
    && typeof detail === 'object'
    && !Array.isArray(detail)
  ) {
    return new ApiError(
      detail.message || `HTTP ${response.status}`,
      response.status,
      detail.code || null,
    )
  }

  if (typeof detail === 'string') {
    return new ApiError(
      detail,
      response.status,
    )
  }

  if (Array.isArray(detail)) {
    const message = detail
      .map((item) => item.msg)
      .filter(Boolean)
      .join('; ')

    return new ApiError(
      message || `HTTP ${response.status}`,
      response.status,
    )
  }

  return new ApiError(
    `HTTP ${response.status}`,
    response.status,
  )
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    headers: {
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    throw await readApiError(response)
  }

  return response.json() as Promise<T>
}

export function getHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>('/api/health')
}

export function getMethods(): Promise<MethodsResponse> {
  return getJson<MethodsResponse>('/api/methods')
}

export function getJob(
  jobId: string,
): Promise<JobStatusResponse> {
  return getJson<JobStatusResponse>(
    `/api/jobs/${encodeURIComponent(jobId)}`,
  )
}

export function getManifest(
  jobId: string,
): Promise<ManifestResponse> {
  return getJson<ManifestResponse>(
    `/api/jobs/${encodeURIComponent(jobId)}/manifest`,
  )
}

export function getJobFileUrl(
  jobId: string,
  filename: string,
): string {
  return (
    `/api/jobs/${encodeURIComponent(jobId)}`
    + `/files/${encodeURIComponent(filename)}`
  )
}

export async function createJob(
  request: CreateJobRequest,
): Promise<JobCreatedResponse> {
  const form = new FormData()
  form.append('image', request.image)
  form.append('method', request.method)
  form.append('vector_mode', request.vectorMode)
  form.append('size', String(request.size))

  const response = await fetch('/api/jobs', {
    method: 'POST',
    body: form,
    headers: {
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    throw await readApiError(response)
  }

  return response.json() as Promise<JobCreatedResponse>
}
