export type IngestStats = {
  pages: number
  text_chunks: number
  images: number
  kg_edges: number
}

export type IngestStatus = {
  status: 'processing' | 'done' | 'error'
  filename: string
  stats: IngestStats | null
  error: string | null
}

export type UploadResponse = {
  task_id: string
  status: string
  filename: string
}

export type SourceResult = {
  global_id?: number
  score?: number
  modality?: string
  source?: string
  page?: number
  text_preview?: string
  image_path?: string
}

export type AskResponse = {
  question: string
  answer_markdown: string
  sources: SourceResult[]
}

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch('/upload-pdf', {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    throw new Error(await readApiError(res, uploadFallback(res.status)))
  }

  return res.json()
}

export async function getIngestStatus(taskId: string): Promise<IngestStatus> {
  const res = await fetch(`/ingest-status/${taskId}`)
  if (!res.ok) {
    throw new Error(await readApiError(res, backendFallback('Status check', res.status)))
  }
  return res.json()
}

export async function askQuestion(question: string, topK = 5): Promise<AskResponse> {
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  })

  if (!res.ok) {
    throw new Error(await readApiError(res, backendFallback('Query', res.status)))
  }

  return res.json()
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const res = await fetch('/health', { method: 'GET' })
    return res.ok
  } catch (_) {
    return false
  }
}

async function readApiError(res: Response, fallback: string): Promise<string> {
  try {
    const contentType = res.headers.get('content-type') ?? ''
    if (!contentType.includes('application/json')) {
      return fallback
    }
    const data = await res.json()
    return data.detail || data.error || fallback
  } catch (_) {
    return fallback
  }
}

function uploadFallback(status: number): string {
  if (status >= 500) {
    return 'Upload failed because the FastAPI backend is not reachable or crashed. Start/restart the backend on port 8000 and try again.'
  }
  return `Upload failed (${status})`
}

function backendFallback(action: string, status: number): string {
  if (status >= 500) {
    return `${action} failed because the FastAPI backend is not reachable or crashed.`
  }
  return `${action} failed (${status})`
}
