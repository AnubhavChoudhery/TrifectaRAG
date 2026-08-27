import { Bot } from 'lucide-react'

export default function TypingIndicator() {
  return (
    <div className="flex gap-3 bg-chat-muted/25 px-4 py-5 md:px-8">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-chat-border bg-chat-surface text-chat-fg">
        <Bot size={16} />
      </div>
      <div className="flex items-center gap-1.5 pt-2">
        <span className="typing-dot" />
        <span className="typing-dot animation-delay-150" />
        <span className="typing-dot animation-delay-300" />
      </div>
    </div>
  )
}
