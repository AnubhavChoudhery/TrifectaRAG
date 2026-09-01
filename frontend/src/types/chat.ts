export type ChatMode = 'agent' | 'general' | 'math' | 'code' | 'study' | 'research'

export type MessageAttachment = {
  type: 'image' | 'table' | 'pdf' | 'docx' | 'file'
  url?: string
  name?: string
  data?: string
}

export type ImagePreview = {
  url: string
  caption?: string
  alt?: string
}

export type Citation = {
  global_id?: number
  score?: number
  modality?: string
  source?: string
  page?: number
  text_preview?: string
  image_path?: string
  retrieval?: string
}

export type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  attachments?: MessageAttachment[]
  imagePreview?: ImagePreview
  toolsUsed?: string[]
  sources?: Citation[]
  timestamp: number
}

export type Conversation = {
  id: string
  title: string
  mode?: ChatMode | null
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
  mode?: ChatMode
  message: string
  attachments?: MessageAttachment[]
}

export type ChatApiResponse = {
  content: string
  imagePreview?: ImagePreview
}

export type CorpusItem = {
  name: string
  kind: 'source' | 'pdf' | string
  indexed: boolean
  enabled: boolean
  pages?: number | null
  chunks: number
  filename?: string | null
}
