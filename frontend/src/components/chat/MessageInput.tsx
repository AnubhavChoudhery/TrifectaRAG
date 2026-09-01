import { FileUp, ImagePlus, Paperclip, Send, Table2 } from 'lucide-react'
import { useCallback, useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react'
import type { ChatMode, MessageAttachment } from '../../types/chat'
import { getModeConfig } from '../../constants/modes'

type MessageInputProps = {
  mode: ChatMode
  disabled?: boolean
  onSend: (message: string, attachments?: MessageAttachment[]) => void
  onUploadPdf?: (file: File) => void
}

function looksLikeMarkdownTable(text: string): boolean {
  const lines = text.trim().split('\n').filter(Boolean)
  if (lines.length < 2) return false
  return lines.every((line) => line.includes('|'))
}

export default function MessageInput({ mode, disabled, onSend, onUploadPdf }: MessageInputProps) {
  const [value, setValue] = useState('')
  const [attachments, setAttachments] = useState<MessageAttachment[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const pdfRef = useRef<HTMLInputElement>(null)
  const placeholder = getModeConfig(mode).placeholder

  const resize = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [])

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return

    let finalAttachments = attachments
    if (looksLikeMarkdownTable(trimmed) && !attachments.some((a) => a.type === 'table')) {
      finalAttachments = [
        ...attachments,
        { type: 'table' as const, data: trimmed, name: 'pasted-table.md' },
      ]
    }

    onSend(trimmed, finalAttachments.length ? finalAttachments : undefined)
    setValue('')
    setAttachments([])
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

  const handleImageUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file?.type.startsWith('image/')) return

    const url = URL.createObjectURL(file)
    setAttachments((prev) => [...prev, { type: 'image', url, name: file.name }])
    e.target.value = ''
  }

  const handlePdfUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.name.toLowerCase().endsWith('.pdf')) {
      onUploadPdf?.(file)
    }
    e.target.value = ''
  }

  const removeAttachment = (index: number) => {
    setAttachments((prev) => {
      const att = prev[index]
      if (att?.type === 'image' && att.url?.startsWith('blob:')) {
        URL.revokeObjectURL(att.url)
      }
      return prev.filter((_, i) => i !== index)
    })
  }

  return (
    <div className="border-t border-chat-border bg-chat-surface px-4 py-4 md:px-8">
      {attachments.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {attachments.map((att, i) => (
            <div
              key={i}
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
          accept="image/*"
          className="hidden"
          onChange={handleImageUpload}
        />
        <input
          ref={pdfRef}
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          onChange={handlePdfUpload}
        />

        {onUploadPdf && (
          <button
            type="button"
            onClick={() => pdfRef.current?.click()}
            disabled={disabled}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-chat-muted-fg transition hover:bg-chat-muted hover:text-chat-fg disabled:opacity-40"
            aria-label="Upload PDF"
            title="Upload a PDF to index"
          >
            <FileUp size={18} />
          </button>
        )}

        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-chat-muted-fg transition hover:bg-chat-muted hover:text-chat-fg disabled:opacity-40"
          aria-label="Upload image"
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
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="max-h-[200px] min-h-[44px] flex-1 resize-none bg-transparent px-1 py-2.5 text-[15px] leading-6 text-chat-fg outline-none placeholder:text-chat-muted-fg disabled:opacity-50"
        />

        <button
          type="button"
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-chat-accent text-white transition hover:bg-chat-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          aria-label="Send message"
        >
          <Send size={18} />
        </button>
      </div>

      <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-chat-muted-fg">
        <Paperclip size={10} className="mr-1 inline" />
        Shift+Enter for new line · Upload a PDF to index · Attach images
      </p>
    </div>
  )
}
