import { useCallback, useEffect, useState } from 'react'
import type { Citation, Conversation, Message, MessageAttachment } from '../types/chat'
import {
  attachFile,
  checkApiHealth,
  getIngestStatus,
  sendAgentChat,
  uploadDocumentFile,
  type VisualHit,
} from '../services/api'
import { sendChatMessage } from '../services/chatApi'

const STORAGE_KEY = 'trifecta-chat-conversations'
const POLL_INTERVAL_MS = 1500
const POLL_TIMEOUT_MS = 10 * 60 * 1000 // 10 minutes for large PDFs

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function hasMarkdownImages(content: string): boolean {
  return /!\[[^\]]*]\(\/image\?path=/.test(content)
}

/** Build the assistant reply from a backend /ask response. */
function buildAssistantMessage(
  content: string,
  sources: Citation[] = [],
  visuals: VisualHit[] = [],
  toolsUsed?: string[],
): Message {
  const attachments: MessageAttachment[] = []
  if (!hasMarkdownImages(content)) {
    for (const visual of visuals) {
      if (!visual.url) continue
      attachments.push({
        type: 'image',
        url: visual.url,
        name: visual.label ?? visual.caption ?? visual.kind ?? 'figure',
      })
    }
    if (!attachments.length) {
      for (const source of sources) {
        if (source.modality !== 'IMAGE' || !source.image_path) continue
        attachments.push({
          type: 'image',
          url: `/image?path=${encodeURIComponent(source.image_path)}`,
          name: `${source.source ?? 'Figure'}${source.page ? ` — page ${source.page}` : ''}`,
        })
      }
    }
  }

  return {
    id: uid(),
    role: 'assistant',
    content,
    attachments: attachments.length ? attachments : undefined,
    toolsUsed: toolsUsed?.length ? toolsUsed : undefined,
    sources: sources.length ? sources : undefined,
    timestamp: Date.now(),
  }
}

function uid() {
  return crypto.randomUUID()
}

function deriveTitle(messages: Message[]): string {
  const first = messages.find((m) => m.role === 'user')
  if (!first) return 'New conversation'
  const text = first.content.trim()
  return text.length > 42 ? `${text.slice(0, 42)}…` : text || 'New conversation'
}

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as Conversation[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function saveConversations(conversations: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
}

function shouldUseAgent(backendUp: boolean): boolean {
  return backendUp
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations)
  const [activeId, setActiveId] = useState<string | null>(() => conversations[0]?.id ?? null)
  const [isLoading, setIsLoading] = useState(false)
  const [engineReady, setEngineReady] = useState(false)
  const [backendUp, setBackendUp] = useState(false)
  const [corpusLabel, setCorpusLabel] = useState<string | null>(null)
  const [indexedChunks, setIndexedChunks] = useState(0)
  const [retrievalLabel, setRetrievalLabel] = useState('HNSW+BM25+KG+MMR')

  useEffect(() => {
    saveConversations(conversations)
  }, [conversations])

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      const health = await checkApiHealth()
      if (cancelled) return
      if (!health) {
        setBackendUp(false)
        return
      }
      setBackendUp(true)
      setEngineReady(health.indexed_chunks > 0)
      setIndexedChunks(health.indexed_chunks)
      setCorpusLabel(health.corpus && health.corpus !== 'empty' ? health.corpus : null)
      if (health.retrieval) setRetrievalLabel(health.retrieval)
    }
    void refresh()
    const timer = window.setInterval(refresh, 8000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null

  const updateConversation = useCallback((id: string, patch: Partial<Conversation>) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, ...patch, updatedAt: Date.now() } : c)),
    )
  }, [])

  const createConversation = useCallback(() => {
    const conv: Conversation = {
      id: uid(),
      title: 'New conversation',
      mode: 'agent',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }
    setConversations((prev) => [conv, ...prev])
    setActiveId(conv.id)
    return conv.id
  }, [])

  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id)
        if (activeId === id) {
          setActiveId(next[0]?.id ?? null)
        }
        return next
      })
    },
    [activeId],
  )

  const selectConversation = useCallback((id: string) => {
    setActiveId(id)
  }, [])

  const sendMessage = useCallback(
    async (
      conversationId: string,
      content: string,
      attachments?: MessageAttachment[],
      files?: File[],
    ) => {
      const conv = conversations.find((c) => c.id === conversationId)
      if (!conv || (!content.trim() && !files?.length && !attachments?.length)) return

      const userMsg: Message = {
        id: uid(),
        role: 'user',
        content: content.trim(),
        attachments,
        timestamp: Date.now(),
      }

      const withUser = [...conv.messages, userMsg]
      updateConversation(conversationId, {
        messages: withUser,
        title: deriveTitle(withUser),
      })

      setIsLoading(true)
      try {
        let assistantMsg: Message
        const useBackend = shouldUseAgent(backendUp)

        if (useBackend) {
          try {
            const parsed = []
            for (const file of files ?? []) {
              parsed.push(await attachFile(file))
            }
            const history = withUser
              .filter((m) => m.role === 'user' || m.role === 'assistant')
              .map((m) => ({ role: m.role, content: m.content }))
            const response = await sendAgentChat('agent', history, content.trim(), parsed)
            assistantMsg = buildAssistantMessage(
              response.answer_markdown,
              response.sources ?? [],
              response.visuals ?? [],
              response.tools_used,
            )
          } catch (err) {
            assistantMsg = {
              id: uid(),
              role: 'assistant',
              content: `I could not finish that request. ${
                err instanceof Error ? err.message : 'The backend was not available.'
              }\n\nKeep \`python api.py\` running on port 8001, and make sure Ollama is running (\`ollama serve\`).`,
              timestamp: Date.now(),
            }
          }
        } else {
          const response = await sendChatMessage('general', content.trim(), attachments)
          assistantMsg = {
            id: uid(),
            role: 'assistant',
            content: response.content,
            imagePreview: response.imagePreview,
            timestamp: Date.now(),
          }
        }
        const full = [...withUser, assistantMsg]
        updateConversation(conversationId, {
          messages: full,
          title: deriveTitle(full),
        })
      } catch (err) {
        const errorMsg: Message = {
          id: uid(),
          role: 'assistant',
          content: `⚠️ ${err instanceof Error ? err.message : 'Something went wrong while answering.'}`,
          timestamp: Date.now(),
        }
        updateConversation(conversationId, { messages: [...withUser, errorMsg] })
      } finally {
        setIsLoading(false)
      }
    },
    [conversations, backendUp, updateConversation],
  )

  const uploadDocument = useCallback(
    async (conversationId: string, file: File) => {
      const conv = conversations.find((c) => c.id === conversationId)
      if (!conv) return

      const userMsg: Message = {
        id: uid(),
        role: 'user',
        content: `📄 Uploaded **${file.name}**`,
        attachments: [{ type: 'pdf', name: file.name }],
        timestamp: Date.now(),
      }
      const statusId = uid()
      const statusMsg: Message = {
        id: statusId,
        role: 'assistant',
        content: `⏳ Reading and indexing **${file.name}**… this can take a moment for large PDFs.`,
        timestamp: Date.now(),
      }

      const baseMessages = [...conv.messages, userMsg, statusMsg]
      updateConversation(conversationId, {
        messages: baseMessages,
        title: conv.messages.length === 0 ? file.name : conv.title,
        documentName: file.name,
        documentIndexed: false,
      })

      const patchStatus = (content: string) =>
        setConversations((prev) =>
          prev.map((c) =>
            c.id === conversationId
              ? {
                  ...c,
                  updatedAt: Date.now(),
                  messages: c.messages.map((m) => (m.id === statusId ? { ...m, content } : m)),
                }
              : c,
          ),
        )

      setIsLoading(true)
      try {
        const { task_id } = await uploadDocumentFile(file)
        const deadline = Date.now() + POLL_TIMEOUT_MS

        while (Date.now() < deadline) {
          await delay(POLL_INTERVAL_MS)
          const status = await getIngestStatus(task_id)

          if (status.status === 'done') {
            const s = status.stats
            const detail = s
              ? ` (${s.pages} pages · ${s.text_chunks} text chunks · ${s.images} images)`
              : ''
            patchStatus(`✅ Indexed **${file.name}**${detail}. Ask me anything about it — I can pull text, tables, graphs, and figures.`)
            updateConversation(conversationId, {
              documentName: file.name,
              documentIndexed: true,
            })
            setEngineReady(true)
            setCorpusLabel(file.name.replace(/\.pdf$/i, ''))
            if (s?.text_chunks) setIndexedChunks((s.text_chunks ?? 0) + (s.images ?? 0))
            return
          }
          if (status.status === 'error') {
            patchStatus(`⚠️ Failed to index **${file.name}**: ${status.error ?? 'unknown error'}`)
            updateConversation(conversationId, { documentIndexed: false })
            return
          }
        }
        patchStatus(`⚠️ Indexing **${file.name}** timed out. Please try again.`)
        updateConversation(conversationId, { documentIndexed: false })
      } catch (err) {
        patchStatus(
          `⚠️ Upload failed: ${err instanceof Error ? err.message : 'unknown error'}`,
        )
        updateConversation(conversationId, { documentIndexed: false })
      } finally {
        setIsLoading(false)
      }
    },
    [conversations, updateConversation],
  )

  return {
    conversations,
    activeConversation,
    activeId,
    isLoading,
    engineReady,
    corpusLabel,
    indexedChunks,
    retrievalLabel,
    createConversation,
    deleteConversation,
    selectConversation,
    sendMessage,
    uploadDocument,
  }
}
