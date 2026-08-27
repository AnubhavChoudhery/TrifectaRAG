import { Menu, MessageSquarePlus, Moon, Sun, Trash2, X } from 'lucide-react'
import type { Conversation } from '../../types/chat'
import { getModeConfig } from '../../constants/modes'

type SidebarProps = {
  conversations: Conversation[]
  activeId: string | null
  isDark: boolean
  isOpen: boolean
  onClose: () => void
  onNewChat: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onToggleTheme: () => void
}

function formatDate(ts: number) {
  const d = new Date(ts)
  const now = new Date()
  const sameDay =
    d.getDate() === now.getDate() &&
    d.getMonth() === now.getMonth() &&
    d.getFullYear() === now.getFullYear()
  if (sameDay) {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export default function Sidebar({
  conversations,
  activeId,
  isDark,
  isOpen,
  onClose,
  onNewChat,
  onSelect,
  onDelete,
  onToggleTheme,
}: SidebarProps) {
  return (
    <>
      {isOpen && (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
          onClick={onClose}
          aria-label="Close sidebar"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[280px] flex-col border-r border-chat-border bg-chat-sidebar transition-transform duration-200 md:static md:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between border-b border-chat-border px-4 py-4">
          <div>
            <h1 className="text-base font-semibold text-chat-fg">Trifecta Tutor</h1>
            <p className="text-xs text-chat-muted-fg">AI learning assistant</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-chat-muted-fg hover:bg-chat-muted md:hidden"
            aria-label="Close menu"
          >
            <X size={18} />
          </button>
        </div>

        <div className="p-3">
          <button
            type="button"
            onClick={() => {
              onNewChat()
              onClose()
            }}
            className="flex w-full items-center gap-2 rounded-xl border border-chat-border bg-chat-surface px-3 py-2.5 text-sm font-medium text-chat-fg transition hover:bg-chat-muted"
          >
            <MessageSquarePlus size={18} />
            New conversation
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-2 pb-2" aria-label="Conversations">
          {conversations.length === 0 ? (
            <p className="px-3 py-6 text-center text-xs leading-5 text-chat-muted-fg">
              No conversations yet. Start a new chat to pick a mode.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {conversations.map((conv) => {
                const modeLabel = conv.mode ? getModeConfig(conv.mode).label : 'Choose mode'
                const isActive = conv.id === activeId
                return (
                  <li key={conv.id} className="group relative">
                    <button
                      type="button"
                      onClick={() => {
                        onSelect(conv.id)
                        onClose()
                      }}
                      className={`flex w-full flex-col rounded-lg px-3 py-2.5 pr-9 text-left transition ${
                        isActive
                          ? 'bg-chat-muted text-chat-fg'
                          : 'text-chat-fg/80 hover:bg-chat-muted/60'
                      }`}
                    >
                      <span className="truncate text-sm font-medium">{conv.title}</span>
                      <span className="mt-0.5 truncate text-[11px] text-chat-muted-fg">
                        {modeLabel} · {formatDate(conv.updatedAt)}
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        onDelete(conv.id)
                      }}
                      className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-chat-muted-fg opacity-0 transition hover:bg-chat-surface hover:text-red-500 group-hover:opacity-100"
                      aria-label="Delete conversation"
                    >
                      <Trash2 size={14} />
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </nav>

        <div className="border-t border-chat-border p-3">
          <button
            type="button"
            onClick={onToggleTheme}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-chat-muted-fg transition hover:bg-chat-muted hover:text-chat-fg"
          >
            {isDark ? <Sun size={16} /> : <Moon size={16} />}
            {isDark ? 'Light mode' : 'Dark mode'}
          </button>
        </div>
      </aside>
    </>
  )
}

export function SidebarToggle({ onOpen }: { onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="rounded-lg p-2 text-chat-muted-fg hover:bg-chat-muted md:hidden"
      aria-label="Open sidebar"
    >
      <Menu size={20} />
    </button>
  )
}
