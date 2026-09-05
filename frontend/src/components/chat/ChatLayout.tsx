import { useEffect, useState } from 'react'
import type { MessageAttachment } from '../../types/chat'
import { useConversations } from '../../hooks/useConversations'
import { useTheme } from '../../hooks/useTheme'
import Sidebar, { SidebarToggle } from './Sidebar'
import ChatWindow from './ChatWindow'
import LibraryDashboard from '../library/LibraryDashboard'

export default function ChatLayout() {
  const {
    conversations,
    activeConversation,
    activeId,
    isLoading,
    engineReady,
    corpusLabel,
    indexedChunks,
    retrievalLabel,
    createConversation,
    deleteConversation,
    selectConversation,
    sendMessage,
    uploadDocument,
  } = useConversations()

  const { isDark, toggleTheme } = useTheme()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [view, setView] = useState<'chat' | 'library'>('chat')

  useEffect(() => {
    if (conversations.length === 0) {
      createConversation()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSend = (message: string, attachments?: MessageAttachment[], files?: File[]) => {
    if (!activeId) return
    sendMessage(activeId, message, attachments, files)
  }

  const handleIndex = (file: File) => {
    if (!activeId) return
    uploadDocument(activeId, file)
  }

  return (
    <div className="flex h-[100dvh] overflow-hidden bg-chat-bg text-chat-fg">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        isDark={isDark}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={() => {
          createConversation()
          setView('chat')
        }}
        onSelect={(id) => {
          selectConversation(id)
          setView('chat')
        }}
        onDelete={deleteConversation}
        onToggleTheme={toggleTheme}
        onOpenLibrary={() => setView('library')}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex shrink-0 items-center gap-2 border-b border-chat-border px-4 py-3 md:hidden">
          <SidebarToggle onOpen={() => setSidebarOpen(true)} />
          <span className="text-sm font-medium">Trifecta Tutor</span>
        </div>

        {view === 'library' ? (
          <LibraryDashboard onBack={() => setView('chat')} />
        ) : (
          <ChatWindow
            conversation={activeConversation}
            isLoading={isLoading}
            onSend={handleSend}
            onIndexDocument={handleIndex}
            engineReady={engineReady}
            corpusLabel={corpusLabel}
            indexedChunks={indexedChunks}
            retrievalLabel={retrievalLabel}
          />
        )}
      </main>
    </div>
  )
}
