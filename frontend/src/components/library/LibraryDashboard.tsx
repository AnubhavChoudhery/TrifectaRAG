import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowLeft, BookOpen, FileUp, Loader2 } from 'lucide-react'
import {
  getIngestStatus,
  ingestKnownCorpus,
  listCorpora,
  toggleCorpus,
  uploadDocumentFile,
  type CorpusItem,
} from '../../services/api'

type LibraryDashboardProps = {
  onBack: () => void
}

export default function LibraryDashboard({ onBack }: LibraryDashboardProps) {
  const [items, setItems] = useState<CorpusItem[]>([])
  const [engineSize, setEngineSize] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    try {
      const data = await listCorpora()
      setItems(data.items)
      setEngineSize(data.engine_size)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load the library.')
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const waitForTask = async (taskId: string) => {
    for (let i = 0; i < 400; i += 1) {
      await new Promise((r) => setTimeout(r, 1500))
      const status = await getIngestStatus(taskId)
      if (status.status === 'done') return
      if (status.status === 'error') {
        throw new Error(status.error ?? 'Indexing failed')
      }
    }
    throw new Error('Indexing timed out')
  }

  const handleToggle = async (item: CorpusItem) => {
    if (!item.indexed) return
    setBusy(item.name)
    try {
      await toggleCorpus(item.name, !item.enabled)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Toggle failed')
    } finally {
      setBusy(null)
    }
  }

  const handleIndex = async (item: CorpusItem) => {
    if (!item.filename) return
    setBusy(item.name)
    try {
      const { task_id } = await ingestKnownCorpus(item.filename)
      await waitForTask(task_id)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Index failed')
    } finally {
      setBusy(null)
    }
  }

  const handleUpload = async (file: File) => {
    setBusy(file.name)
    try {
      const { task_id } = await uploadDocumentFile(file)
      await waitForTask(task_id)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-chat-border px-4 py-3 md:px-8">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="rounded-lg p-1.5 text-chat-muted-fg hover:bg-chat-muted hover:text-chat-fg"
            aria-label="Back to chat"
          >
            <ArrowLeft size={18} />
          </button>
          <div>
            <h2 className="text-sm font-semibold text-chat-fg">Library</h2>
            <p className="text-[11px] text-chat-muted-fg">
              {engineSize} chunks in the live index · hybrid HNSW + BM25 + KG + MMR
            </p>
          </div>
        </div>
        <div>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx,.txt,.md,.csv,application/pdf"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              e.target.value = ''
              if (file) void handleUpload(file)
            }}
          />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={Boolean(busy)}
            className="inline-flex items-center gap-2 rounded-xl bg-chat-accent px-3 py-2 text-sm font-medium text-white hover:bg-chat-accent-hover disabled:opacity-40"
          >
            <FileUp size={16} />
            Index a file
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
        <p className="mb-4 max-w-2xl text-sm leading-6 text-chat-muted-fg">
          Turn sources on or off for retrieval. Index another textbook or notes (PDF, DOCX, TXT)
          to add a RAG database without leaving chat.
        </p>
        {error && <p className="mb-4 text-sm text-red-400">{error}</p>}
        {busy && (
          <p className="mb-4 inline-flex items-center gap-2 text-sm text-chat-muted-fg">
            <Loader2 size={14} className="animate-spin" />
            Working on {busy}…
          </p>
        )}
        <ul className="space-y-2">
          {items.map((item) => (
            <li
              key={`${item.kind}-${item.name}`}
              className="flex items-center justify-between gap-4 rounded-xl border border-chat-border bg-chat-surface px-4 py-3"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <BookOpen size={16} className="shrink-0 text-chat-accent" />
                  <span className="truncate text-sm font-medium text-chat-fg">{item.name}</span>
                </div>
                <p className="mt-1 text-[11px] text-chat-muted-fg">
                  {item.indexed
                    ? `${item.pages ?? 0} pages · ${item.chunks} chunks · ${item.enabled ? 'searching' : 'paused'}`
                    : `On disk${item.filename ? ` · ${item.filename}` : ''} · not indexed yet`}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {item.indexed ? (
                  <button
                    type="button"
                    onClick={() => void handleToggle(item)}
                    disabled={Boolean(busy)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium ${
                      item.enabled
                        ? 'bg-chat-accent/15 text-chat-accent'
                        : 'bg-chat-muted text-chat-muted-fg'
                    }`}
                  >
                    {item.enabled ? 'On' : 'Off'}
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => void handleIndex(item)}
                    disabled={Boolean(busy)}
                    className="rounded-lg bg-chat-muted px-3 py-1.5 text-xs font-medium text-chat-fg hover:bg-chat-muted/80"
                  >
                    Create RAG DB
                  </button>
                )}
              </div>
            </li>
          ))}
          {items.length === 0 && !error && (
            <li className="text-sm text-chat-muted-fg">No sources yet. Index a PDF to start.</li>
          )}
        </ul>
      </div>
    </div>
  )
}
