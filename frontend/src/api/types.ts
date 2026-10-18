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
