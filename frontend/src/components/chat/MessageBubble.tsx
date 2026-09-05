import { Bot, User } from 'lucide-react'
import type { Message } from '../../types/chat'
import MarkdownRenderer from './MarkdownRenderer'
import ImagePreviewCard from './ImagePreview'
import { AttachmentList } from './AttachmentList'

type MessageBubbleProps = {
  message: Message
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div
      className={`flex gap-3 px-4 py-5 md:px-8 ${isUser ? 'bg-transparent' : 'bg-chat-muted/25'}`}
    >
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
          isUser
            ? 'bg-chat-accent text-white'
            : 'border border-chat-border bg-chat-surface text-chat-fg'
        }`}
        aria-hidden
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      <div className="min-w-0 flex-1 space-y-2 pt-0.5">
        <p className="text-xs font-semibold uppercase tracking-wide text-chat-muted-fg">
          {isUser ? 'You' : 'Assistant'}
        </p>

        {isUser ? (
          <div className="whitespace-pre-wrap text-[15px] leading-7 text-chat-fg">{message.content}</div>
        ) : (
          <MarkdownRenderer content={message.content} />
        )}

        <AttachmentList attachments={message.attachments} />

        {message.imagePreview && (
          <div className="pt-2">
            <ImagePreviewCard preview={message.imagePreview} />
          </div>
        )}

        {message.sources && message.sources.length > 0 && (
          <details className="pt-2">
            <summary className="cursor-pointer text-[11px] text-chat-muted-fg">
              Sources · {message.sources.length} · {message.sources[0]?.retrieval ?? 'hybrid HNSW+BM25+KG+MMR'}
            </summary>
            <ul className="mt-2 space-y-1.5 text-[12px] leading-5 text-chat-muted-fg">
              {message.sources.slice(0, 8).map((src, i) => (
                <li key={`${src.global_id ?? i}-${src.page ?? 0}`}>
                  <span className="font-medium text-chat-fg/80">{src.source ?? 'PDF'}</span>
                  {src.page != null ? ` p.${src.page}` : ''}
                  {src.global_id != null ? ` · gid ${src.global_id}` : ''}
                  {src.score != null ? ` · rrf ${src.score.toFixed(3)}` : ''}
                  {src.text_preview ? ` — ${src.text_preview.slice(0, 140)}` : ''}
                </li>
              ))}
            </ul>
          </details>
        )}

        {message.toolsUsed && message.toolsUsed.length > 0 && (
          <p className="pt-1 text-[11px] text-chat-muted-fg">
            Used {message.toolsUsed.join(' · ')}
          </p>
        )}
      </div>
    </div>
  )
}
