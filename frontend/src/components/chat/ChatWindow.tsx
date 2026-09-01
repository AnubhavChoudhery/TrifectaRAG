import { useEffect, useRef } from 'react'
import type { Conversation, MessageAttachment } from '../../types/chat'
import MessageBubble from './MessageBubble'
import MessageInput from './MessageInput'
import TypingIndicator from './TypingIndicator'

type ChatWindowProps = {
  conversation: Conversation | null
  isLoading: boolean
  onSend: (message: string, attachments?: MessageAttachment[], files?: File[]) => void
  onIndexDocument: (file: File) => void
  engineReady?: boolean
  corpusLabel?: string | null
  indexedChunks?: number
}

export default function ChatWindow({
  conversation,
  isLoading,
  onSend,
  onIndexDocument,
  engineReady = false,
  corpusLabel = null,
  indexedChunks = 0,
}: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation?.messages, isLoading])

  if (!conversation) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
        <h2 className="text-xl font-semibold text-chat-fg">Welcome to Trifecta Tutor</h2>
        <p className="mt-2 max-w-md text-sm leading-6 text-chat-muted-fg">
          Create a new conversation from the sidebar to begin.
        </p>
      </div>
    )
  }

  const hasMessages = conversation.messages.length > 0

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-chat-border px-4 py-3 md:px-8">
        <div>
          <p className="text-sm font-semibold text-chat-fg">Trifecta Tutor</p>
          <p className="text-[11px] text-chat-muted-fg">
            One agent · study, math, code, and web research as needed
          </p>
        </div>
        <div className="hidden min-w-0 flex-col items-end sm:flex">
          <span className="max-w-sm truncate text-xs text-chat-muted-fg">{conversation.title}</span>
          {engineReady && corpusLabel ? (
            <span className="max-w-sm truncate text-[11px] text-chat-muted-fg">
              Library · {corpusLabel}
              {indexedChunks ? ` · ${indexedChunks} chunks` : ''}
            </span>
          ) : (
            <span className="text-[11px] text-chat-muted-fg">Open Library to index a source</span>
          )}
        </div>
      </header>

      <div className="flex-1 overflow-y-auto overscroll-contain">
        {!hasMessages && (
          <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
            <p className="max-w-lg text-sm leading-6 text-chat-muted-fg">
              Ask a practice question, attach a worksheet image, or drop a PDF/DOCX. I will
              search the library, pull figures when they help, and look online when the notes
              are not enough.
            </p>
          </div>
        )}

        {conversation.messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {isLoading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      <MessageInput disabled={isLoading} onSend={onSend} onIndexDocument={onIndexDocument} />
    </div>
  )
}
