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
  retrieval?: string
}

export type VisualHit = {
  url: string
  caption?: string
  kind?: 'figure' | 'graph' | 'table' | string
  page?: number
  source?: string
  label?: string
}

export type HealthStatus = {
  status: string
  engine_ready: boolean
  indexed_chunks: number
  page_count?: number
  corpus?: string
  ollama_model?: string
}

export type AskResponse = {
  question: string
  answer_markdown: string
  sources: SourceResult[]
  visuals?: VisualHit[]
  tools_used?: string[]
  used_agent?: boolean
  engine_size?: number
  corpus?: string
}

export type AttachResult = {
  type: string
  name: string
  text: string
  image_path?: string | null
  similar?: string
}

export type CorpusItem = {
  name: string
  kind: string
  indexed: boolean
  enabled: boolean
  pages?: number | null
  chunks: number
  filename?: string | null
}

export type CorporaResponse = {
  items: CorpusItem[]
  snapshots: { name: string; filename: string }[]
  engine_size: number
  corpus: string
}

export type ChatTurn = {
  role: 'user' | 'assistant' | string
  content: string
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
    body: JSON.stringify({ question, top_k: topK, mode: 'study' }),
  })

  if (!res.ok) {
    throw new Error(await readApiError(res, backendFallback('Query', res.status)))
  }

  return res.json()
}

export async function sendAgentChat(
  mode: string,
  messages: ChatTurn[],
  question?: string,
  attachments?: AttachResult[],
): Promise<AskResponse> {
  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: mode || 'agent', messages, question, attachments }),
  })

  if (!res.ok) {
    throw new Error(await readApiError(res, backendFallback('Chat', res.status)))
  }

  return res.json()
}

export async function attachFile(file: File): Promise<AttachResult> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch('/attach', { method: 'POST', body: formData })
  if (!res.ok) {
    throw new Error(await readApiError(res, backendFallback('Attach', res.status)))
  }
  return res.json()
}

export async function uploadDocumentFile(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch('/upload', { method: 'POST', body: formData })
  if (!res.ok) {
    throw new Error(await readApiError(res, uploadFallback(res.status)))
  }
  return res.json()
}

export async function listCorpora(): Promise<CorporaResponse> {
  const res = await fetch('/corpora')
  if (!res.ok) {
    throw new Error(await readApiError(res, backendFallback('Library', res.status)))
  }
  return res.json()
}

export async function toggleCorpus(name: string, enabled: boolean): Promise<void> {
  const res = await fetch('/corpora/toggle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, enabled }),
  })
  if (!res.ok) {
    throw new Error(await readApiError(res, backendFallback('Toggle corpus', res.status)))
  }
}

export async function ingestKnownCorpus(filename: string): Promise<UploadResponse> {
  const res = await fetch('/corpora/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename }),
  })
  if (!res.ok) {
    throw new Error(await readApiError(res, backendFallback('Ingest', res.status)))
  }
  return res.json()
}

export async function checkApiHealth(): Promise<HealthStatus | null> {
  try {
    const res = await fetch('/health', { method: 'GET' })
    if (!res.ok) return null
    return (await res.json()) as HealthStatus
  } catch (_) {
    return null
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
    return 'Upload failed because the FastAPI backend is not reachable or crashed. Start/restart the backend on port 8001 and try again.'
  }
  return `Upload failed (${status})`
}

function backendFallback(action: string, status: number): string {
  if (status >= 500) {
    return `${action} failed because the FastAPI backend is not reachable or crashed.`
  }
  return `${action} failed (${status})`
}
