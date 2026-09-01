import type { ChatMode } from '../../types/chat'
import { CHAT_MODES } from '../../constants/modes'

type ModeSelectorProps = {
  onSelect: (mode: ChatMode) => void
}

export default function ModeSelector({ onSelect }: ModeSelectorProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 py-12">
      <div className="mb-10 max-w-lg text-center">
        <h2 className="text-2xl font-semibold text-chat-fg md:text-3xl">
          Choose a mode to start learning
        </h2>
        <p className="mt-3 text-sm leading-6 text-chat-muted-fg md:text-base">
          Pick how you'd like this conversation to work. You can switch modes anytime from the
          chat header.
        </p>
      </div>

      <div className="grid w-full max-w-3xl gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {CHAT_MODES.map((mode) => (
          <button
            key={mode.id}
            type="button"
            onClick={() => onSelect(mode.id)}
            className="group flex flex-col rounded-2xl border border-chat-border bg-chat-surface p-5 text-left shadow-sm transition hover:border-chat-accent/40 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-chat-accent"
          >
            <span
              className={`mb-4 inline-flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br ${mode.accent} text-lg font-semibold text-white shadow-sm`}
            >
              {mode.icon}
            </span>
            <span className="text-base font-semibold text-chat-fg">{mode.label}</span>
            <span className="mt-1.5 text-sm leading-6 text-chat-muted-fg">{mode.description}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
