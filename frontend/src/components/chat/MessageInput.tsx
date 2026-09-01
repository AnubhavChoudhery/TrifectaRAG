import { FileUp, ImagePlus, Paperclip, Send, Table2 } from 'lucide-react'
import { useCallback, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import type { MessageAttachment } from '../../types/chat'

type MessageInputProps = {
  disabled?: boolean
  onSend: (message: string, attachments?: MessageAttachment[], files?: File[]) => void
  onIndexDocument?: (file: File) => void
}

function looksLikeMarkdownTable(text: string): boolean {
  const lines = text.trim().split('\n').filter(Boolean)
  if (lines.length < 2) return false
  return lines.every((line) => line.includes('|'))
}

export default function MessageInput({ disabled, onSend, onIndexDocument }: MessageInputProps) {
  const [value, setValue] = useState('')
  const [attachments, setAttachments] = useState<MessageAttachment[]>([])
  const pendingFiles = useRef<File[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const indexRef = useRef<HTMLInputElement>(null)

  const resize = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [])

  const handleSend = () => {
    const trimmed = value.trim()
    const files = pendingFiles.current
    if (disabled) return
    if (!trimmed && files.length === 0 && attachments.length === 0) return

    let finalAttachments = attachments
    if (looksLikeMarkdownTable(trimmed) && !attachments.some((a) => a.type === 'table')) {
      finalAttachments = [
        ...attachments,
        { type: 'table' as const, data: trimmed, name: 'pasted-table.md' },
      ]
    }

    onSend(
      trimmed || (files.length ? 'Please analyze the attached file(s).' : ''),
      finalAttachments.length ? finalAttachments : undefined,
      files.length ? files : undefined,
    )
    setValue('')
    setAttachments([])
    pendingFiles.current = []
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const addFiles = (list: FileList | File[]) => {
    const next: MessageAttachment[] = []
    for (const file of Array.from(list)) {
      pendingFiles.current = [...pendingFiles.current, file]
      const lower = file.name.toLowerCase()
      if (file.type.startsWith('image/')) {
        next.push({ type: 'image', url: URL.createObjectURL(file), name: file.name })
      } else if (lower.endsWith('.pdf')) {
        next.push({ type: 'pdf', name: file.name })
      } else if (lower.endsWith('.docx')) {
        next.push({ type: 'docx', name: file.name })
      } else {
        next.push({ type: 'file', name: file.name })
      }
    }
    setAttachments((prev) => [...prev, ...next])
  }

  const handleAttach = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) addFiles(e.target.files)
    e.target.value = ''
  }

  const handleIndex = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (file) onIndexDocument?.(file)
  }

  const removeAttachment = (index: number) => {
    setAttachments((prev) => {
      const att = prev[index]
      if (att?.type === 'image' && att.url?.startsWith('blob:')) {
        URL.revokeObjectURL(att.url)
      }
      pendingFiles.current = pendingFiles.current.filter((_, i) => i !== index)
      return prev.filter((_, i) => i !== index)
    })
  }

  const canSend = Boolean(value.trim() || pendingFiles.current.length || attachments.length)

  return (
    <div className="border-t border-chat-border bg-chat-surface px-4 py-4 md:px-8">
      {attachments.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {attachments.map((att, i) => (
            <div
              key={`${att.name}-${i}`}
              className="relative flex items-center gap-2 rounded-lg border border-chat-border bg-chat-muted/40 px-2 py-1.5 text-xs"
            >
              {att.type === 'image' && att.url ? (
                <img src={att.url} alt="" className="h-8 w-8 rounded object-cover" />
              ) : (
                <Table2 size={14} />
              )}
              <span className="max-w-[120px] truncate">{att.name ?? att.type}</span>
              <button
                type="button"
                onClick={() => removeAttachment(i)}
                className="ml-1 rounded px-1 text-chat-muted-fg hover:bg-chat-muted hover:text-chat-fg"
                aria-label="Remove attachment"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-chat-border bg-chat-muted/30 p-2 shadow-sm focus-within:border-chat-accent/50 focus-within:ring-2 focus-within:ring-chat-accent/20">
        <input
          ref={fileRef}
          type="file"
          accept="image/*,.pdf,.docx,.txt,.md,.csv,application/pdf"
          multiple
          className="hidden"
          onChange={handleAttach}
        />
        <input
          ref={indexRef}
          type="file"
          accept=".pdf,.docx,.txt,.md,.csv,application/pdf"
          className="hidden"
          onChange={handleIndex}
        />

        {onIndexDocument && (
          <button
            type="button"
            onClick={() => indexRef.current?.click()}
            disabled={disabled}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-chat-muted-fg transition hover:bg-chat-muted hover:text-chat-fg disabled:opacity-40"
            aria-label="Index a document into the library"
            title="Index a PDF or DOCX into the RAG library"
          >
            <FileUp size={18} />
          </button>
        )}

        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-chat-muted-fg transition hover:bg-chat-muted hover:text-chat-fg disabled:opacity-40"
          aria-label="Attach file"
          title="Attach image, PDF, or DOCX for this question"
        >
          <ImagePlus size={18} />
        </button>

        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => {
            setValue(e.target.value)
            resize()
          }}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything — attach a worksheet or notes if you have them…"
          disabled={disabled}
          rows={1}
          className="max-h-[200px] min-h-[44px] flex-1 resize-none bg-transparent px-1 py-2.5 text-[15px] leading-6 text-chat-fg outline-none placeholder:text-chat-muted-fg disabled:opacity-50"
        />

        <button
          type="button"
          onClick={handleSend}
          disabled={disabled || !canSend}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-chat-accent text-white transition hover:bg-chat-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          aria-label="Send message"
        >
          <Send size={18} />
        </button>
      </div>

      <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-chat-muted-fg">
        <Paperclip size={10} className="mr-1 inline" />
        Shift+Enter for a new line · Attach for this turn · File-up indexes a RAG source
      </p>
    </div>
  )
}
