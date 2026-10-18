import type {
  HealthResponse,
  MethodsResponse,
} from './types'

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    headers: {
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    throw new Error(
      `${path} returned HTTP ${response.status}`,
    )
  }

  return response.json() as Promise<T>
}

export function getHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>('/api/health')
}

export function getMethods(): Promise<MethodsResponse> {
  return getJson<MethodsResponse>('/api/methods')
}
