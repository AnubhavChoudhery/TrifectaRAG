export type ChatMode = 'general' | 'math' | 'code' | 'study' | 'research'

export type MessageAttachment = {
  type: 'image' | 'table' | 'pdf'
  url?: string
  name?: string
  /** Raw markdown table or tabular text */
  data?: string
}

export type ImagePreview = {
  url: string
  caption?: string
  alt?: string
}

export type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  attachments?: MessageAttachment[]
  imagePreview?: ImagePreview
  timestamp: number
}

export type Conversation = {
  id: string
  title: string
  mode: ChatMode | null
  messages: Message[]
  documentName?: string
  documentIndexed?: boolean
  createdAt: number
  updatedAt: number
}

export type ChatModeConfig = {
  id: ChatMode
  label: string
  description: string
  icon: string
  placeholder: string
  accent: string
}

export type SendMessagePayload = {
  conversationId: string
  mode: ChatMode
  message: string
  attachments?: MessageAttachment[]
}

export type ChatApiResponse = {
  content: string
  imagePreview?: ImagePreview
}
