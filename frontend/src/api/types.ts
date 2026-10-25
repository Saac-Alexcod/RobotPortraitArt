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

export type JobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'

export interface CreateJobRequest {
  image: File
  method: string
  vectorMode: VectorMode
  size: number
}

export interface JobCreatedResponse {
  job_id: string
  status: JobStatus
  method: string
  vector_mode: VectorMode
  size: number
}

export interface JobErrorDetail {
  type: string
  message: string
}

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  method: string
  vector_mode: VectorMode
  size: number
  original_filename: string | null
  input_file: string
  created_at: string
  updated_at: string | null
  started_at: string | null
  completed_at: string | null
  error: JobErrorDetail | null
  result_files: Record<string, string>
}

export interface VectorizationMetrics {
  width?: number
  height?: number
  path_count?: number
  stroke_count?: number
  raw_stroke_count?: number
  merge_count?: number
  planning_mode?: string
  ordering_mode?: string
  pen_down_length_px?: number
  pen_up_length_px?: number
  pen_up_greedy_px?: number
  two_opt_improvement_px?: number
  two_opt_improvement_ratio?: number
  connected_component_count?: number
  theoretical_min_trail_count?: number
  edge_reuse_count?: number
}

export interface TimingMetrics {
  preprocess?: number
  generate?: number
  vectorize?: number
  rasterize?: number
  total?: number
}

export interface ManifestResponse {
  evaluation?: {
    protocol: string
    reference: string
    note: string
    source_to_portrait: { ssim: number; edge_f1: number | null }
    source_to_vector: { ssim: number; edge_f1: number | null }
    vector_retention: { ssim: number; edge_f1: number | null }
  }
  schema_version?: number
  status?: string
  method?: string
  files?: Record<string, string>
  vectorization?: VectorizationMetrics
  timings_ms?: TimingMetrics
  warnings?: string[]
}
