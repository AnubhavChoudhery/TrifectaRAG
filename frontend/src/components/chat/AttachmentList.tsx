import { FileText } from 'lucide-react'
import type { MessageAttachment } from '../../types/chat'
import MarkdownRenderer from './MarkdownRenderer'

type AttachmentListProps = {
  attachments?: MessageAttachment[]
}

export function AttachmentList({ attachments }: AttachmentListProps) {
  if (!attachments?.length) return null

  return (
    <div className="mt-2 flex flex-col gap-2">
      {attachments.map((att, i) => {
        if (att.type === 'image' && att.url) {
          return (
            <img
              key={i}
              src={att.url}
              alt={att.name ?? 'Uploaded image'}
              className="max-h-64 max-w-full rounded-xl border border-chat-border object-contain"
            />
          )
        }
        if (att.type === 'pdf' || att.type === 'docx' || att.type === 'file') {
          return (
            <div
              key={i}
              className="flex items-center gap-2 self-start rounded-xl border border-chat-border bg-chat-muted/30 px-3 py-2 text-sm text-chat-fg"
            >
              <FileText size={16} className="shrink-0 text-chat-accent" />
              <span className="max-w-[240px] truncate">{att.name ?? 'document'}</span>
            </div>
          )
        }
        if (att.type === 'table' && att.data) {
          return (
            <div key={i} className="rounded-xl border border-chat-border bg-chat-muted/30 p-3">
              <MarkdownRenderer content={att.data} />
            </div>
          )
        }
        return null
      })}
    </div>
  )
}
