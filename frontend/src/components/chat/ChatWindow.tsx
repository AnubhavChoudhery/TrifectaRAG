import { useEffect, useRef } from 'react'
import type { ChatMode, Conversation, MessageAttachment } from '../../types/chat'
import { getModeConfig } from '../../constants/modes'
import ModeSelector from './ModeSelector'
import ModeBadge from './ModeBadge'
import MessageBubble from './MessageBubble'
import MessageInput from './MessageInput'
import TypingIndicator from './TypingIndicator'

type ChatWindowProps = {
  conversation: Conversation | null
  isLoading: boolean
  onSelectMode: (mode: ChatMode) => void
  onChangeMode: (mode: ChatMode) => void
  onSend: (message: string, attachments?: MessageAttachment[]) => void
  onUploadPdf: (file: File) => void
}

export default function ChatWindow({
  conversation,
  isLoading,
  onSelectMode,
  onChangeMode,
  onSend,
  onUploadPdf,
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

  const needsMode = conversation.mode === null
  const hasMessages = conversation.messages.length > 0
  const modeConfig = conversation.mode ? getModeConfig(conversation.mode) : null

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {!needsMode && conversation.mode && (
        <header className="flex shrink-0 items-center justify-between gap-3 border-b border-chat-border px-4 py-3 md:px-8">
          <ModeBadge mode={conversation.mode} onChange={onChangeMode} />
          <div className="hidden min-w-0 flex-col items-end sm:flex">
            <span className="max-w-sm truncate text-xs text-chat-muted-fg">{conversation.title}</span>
            {conversation.documentName && (
              <span className="max-w-sm truncate text-[11px] text-chat-muted-fg">
                {conversation.documentIndexed ? 'PDF ready' : 'Indexing PDF'} · {conversation.documentName}
              </span>
            )}
          </div>
        </header>
      )}

      <div className="flex-1 overflow-y-auto overscroll-contain">
        {needsMode ? (
          <ModeSelector onSelect={onSelectMode} />
        ) : (
          <>
            {!hasMessages && (
              <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
                <p className="text-sm text-chat-muted-fg">
                  {modeConfig?.id === 'study' || modeConfig?.id === 'research'
                    ? 'Upload a PDF or ask a question to start.'
                    : `You're in ${modeConfig?.label}. Send a message to begin.`}
                </p>
              </div>
            )}

            {conversation.messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {isLoading && <TypingIndicator />}
            <div ref={bottomRef} />
          </>
        )}
      </div>

      {conversation.mode && (
        <MessageInput
          mode={conversation.mode}
          disabled={isLoading}
          onSend={onSend}
          onUploadPdf={onUploadPdf}
        />
      )}
    </div>
  )
}
