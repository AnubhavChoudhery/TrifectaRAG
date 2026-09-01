import { useEffect, useState } from 'react'
import type { ChatMode, MessageAttachment } from '../../types/chat'
import { useConversations } from '../../hooks/useConversations'
import { useTheme } from '../../hooks/useTheme'
import Sidebar, { SidebarToggle } from './Sidebar'
import ChatWindow from './ChatWindow'

export default function ChatLayout() {
  const {
    conversations,
    activeConversation,
    activeId,
    isLoading,
    createConversation,
    deleteConversation,
    selectConversation,
    setMode,
    sendMessage,
    uploadDocument,
  } = useConversations()

  const { isDark, toggleTheme } = useTheme()
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    if (conversations.length === 0) {
      createConversation()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSelectMode = (mode: ChatMode) => {
    if (!activeId) return
    setMode(activeId, mode)
  }

  const handleChangeMode = (mode: ChatMode) => {
    if (!activeId) return
    setMode(activeId, mode)
  }

  const handleSend = (message: string, attachments?: MessageAttachment[]) => {
    if (!activeId) return
    sendMessage(activeId, message, attachments)
  }

  const handleUploadPdf = (file: File) => {
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
        onNewChat={createConversation}
        onSelect={selectConversation}
        onDelete={deleteConversation}
        onToggleTheme={toggleTheme}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex shrink-0 items-center gap-2 border-b border-chat-border px-4 py-3 md:hidden">
          <SidebarToggle onOpen={() => setSidebarOpen(true)} />
          <span className="text-sm font-medium">Trifecta Tutor</span>
        </div>

        <ChatWindow
          conversation={activeConversation}
          isLoading={isLoading}
          onSelectMode={handleSelectMode}
          onChangeMode={handleChangeMode}
          onSend={handleSend}
          onUploadPdf={handleUploadPdf}
        />
      </main>
    </div>
  )
}
