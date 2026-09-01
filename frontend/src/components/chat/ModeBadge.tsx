import { ChevronDown, Sparkles } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { ChatMode } from '../../types/chat'
import { CHAT_MODES, getModeConfig } from '../../constants/modes'

type ModeBadgeProps = {
  mode: ChatMode
  onChange: (mode: ChatMode) => void
}

export default function ModeBadge({ mode, onChange }: ModeBadgeProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const config = getModeConfig(mode)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-2 rounded-full border border-chat-border bg-chat-muted/50 px-3 py-1.5 text-sm font-medium text-chat-fg transition hover:bg-chat-muted"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <Sparkles size={14} className="text-chat-accent" />
        <span>{config.label}</span>
        <ChevronDown size={14} className={`transition ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <ul
          role="listbox"
          className="absolute left-0 top-full z-20 mt-2 min-w-[220px] overflow-hidden rounded-xl border border-chat-border bg-chat-surface py-1 shadow-lg"
        >
          {CHAT_MODES.map((m) => (
            <li key={m.id}>
              <button
                type="button"
                role="option"
                aria-selected={m.id === mode}
                onClick={() => {
                  onChange(m.id)
                  setOpen(false)
                }}
                className={`flex w-full items-center gap-3 px-3 py-2.5 text-left text-sm transition hover:bg-chat-muted ${
                  m.id === mode ? 'bg-chat-accent/10 font-medium text-chat-accent' : 'text-chat-fg'
                }`}
              >
                <span className="w-5 text-center">{m.icon}</span>
                {m.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
