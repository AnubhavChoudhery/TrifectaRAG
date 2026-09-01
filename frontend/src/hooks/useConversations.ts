import { useCallback, useEffect, useState } from 'react'
import type { ChatMode, Conversation, Message, MessageAttachment } from '../types/chat'
import { askQuestion, getIngestStatus, uploadPdf } from '../services/api'
import { sendChatMessage } from '../services/chatApi'

const STORAGE_KEY = 'trifecta-chat-conversations'
const POLL_INTERVAL_MS = 1500
const POLL_TIMEOUT_MS = 10 * 60 * 1000 // 10 minutes for large PDFs

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/** Build the assistant reply from a backend /ask response. */
function buildAssistantMessage(content: string, sources: { modality?: string; image_path?: string; page?: number; source?: string }[]): Message {
  const firstImage = sources.find((s) => s.modality === 'IMAGE' && s.image_path)
  return {
    id: uid(),
    role: 'assistant',
    content,
    imagePreview: firstImage
      ? {
          url: `/image?path=${encodeURIComponent(firstImage.image_path as string)}`,
          caption: `${firstImage.source ?? 'Figure'}${firstImage.page ? ` — page ${firstImage.page}` : ''}`,
        }
      : undefined,
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

function shouldUseDocumentRetrieval(message: string): boolean {
  return /\b(pdf|document|uploaded|textbook|source|page|cite|citation|according to|from the file|from the notes)\b/i.test(
    message,
  )
}

function shouldQueryBackend(mode: ChatMode, hasIndexedDocument: boolean, message: string): boolean {
  if (!hasIndexedDocument) return false
  if (mode === 'study') return true
  if (mode === 'research') return true
  if (mode === 'general') return shouldUseDocumentRetrieval(message)
  return false
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations)
  const [activeId, setActiveId] = useState<string | null>(() => conversations[0]?.id ?? null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    saveConversations(conversations)
  }, [conversations])

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
      mode: null,
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

  const setMode = useCallback(
    (id: string, mode: ChatMode) => {
      updateConversation(id, { mode })
    },
    [updateConversation],
  )

  const sendMessage = useCallback(
    async (conversationId: string, content: string, attachments?: MessageAttachment[]) => {
      const conv = conversations.find((c) => c.id === conversationId)
      if (!conv?.mode || !content.trim()) return

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
        const useBackend = shouldQueryBackend(conv.mode, Boolean(conv.documentIndexed), content)

        if (useBackend) {
          try {
            const response = await askQuestion(content.trim())
            assistantMsg = buildAssistantMessage(response.answer_markdown, response.sources ?? [])
          } catch (err) {
            if (conv.mode === 'study') {
              assistantMsg = {
                id: uid(),
                role: 'assistant',
                content: `I could not answer from the PDF right now. ${
                  err instanceof Error ? err.message : 'The document backend was not available.'
                }\n\nPlease make sure the backend is running on port 8000, then upload the PDF again in this conversation and wait until it says **PDF ready**.`,
                timestamp: Date.now(),
              }
            } else {
            const response = await sendChatMessage(conv.mode, content.trim(), attachments)
            assistantMsg = {
              id: uid(),
              role: 'assistant',
              content: response.content,
              imagePreview: response.imagePreview,
              timestamp: Date.now(),
            }
            }
          }
        } else {
          const response = await sendChatMessage(conv.mode, content.trim(), attachments)
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
    [conversations, updateConversation],
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
        const { task_id } = await uploadPdf(file)
        const deadline = Date.now() + POLL_TIMEOUT_MS

        while (Date.now() < deadline) {
          await delay(POLL_INTERVAL_MS)
          const status = await getIngestStatus(task_id)

          if (status.status === 'done') {
            const s = status.stats
            const detail = s
              ? ` (${s.pages} pages · ${s.text_chunks} text chunks · ${s.images} images)`
              : ''
            patchStatus(`✅ Indexed **${file.name}**${detail}. Ask me anything about it.`)
            updateConversation(conversationId, {
              documentName: file.name,
              documentIndexed: true,
            })
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
    createConversation,
    deleteConversation,
    selectConversation,
    setMode,
    sendMessage,
    uploadDocument,
  }
}
