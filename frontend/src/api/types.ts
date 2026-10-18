export interface HealthResponse {
  status: string
  service: string
}

export interface MethodStatus {
  name: string
  available: boolean
  reason: string | null
  details: Record<string, unknown>
}

export interface MethodsResponse {
  methods: MethodStatus[]
}

export type VectorMode = 'contour' | 'centerline'

export interface CreateJobRequest {
  image: File
  method: string
  vectorMode: VectorMode
  size: number
}

export interface JobCreatedResponse {
  job_id: string
  status: string
  method: string
  vector_mode: VectorMode
  size: number
}
