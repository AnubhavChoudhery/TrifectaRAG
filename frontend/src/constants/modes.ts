import type { ChatModeConfig } from '../types/chat'

export const CHAT_MODES: ChatModeConfig[] = [
  {
    id: 'general',
    label: 'General Chat',
    description: 'Open-ended chat; can search notes or the web when needed.',
    icon: '💬',
    placeholder: 'Ask anything…',
    accent: 'from-violet-500 to-purple-600',
  },
  {
    id: 'math',
    label: 'Math Solver',
    description: 'Step-by-step solutions; pulls textbook notes and figures when relevant.',
    icon: '∑',
    placeholder: 'Enter a math problem…',
    accent: 'from-blue-500 to-cyan-600',
  },
  {
    id: 'code',
    label: 'Code Helper',
    description: 'Debug, explain, and write code with examples.',
    icon: '{ }',
    placeholder: 'Describe your coding question…',
    accent: 'from-emerald-500 to-teal-600',
  },
  {
    id: 'study',
    label: 'Study Tutor',
    description: 'Exam-style tutor that retrieves textbook text, figures, and tables.',
    icon: '📚',
    placeholder: 'What would you like to study?',
    accent: 'from-amber-500 to-orange-600',
  },
  {
    id: 'research',
    label: 'Research Assistant',
    description: 'Search local PDFs and the web; cite sources.',
    icon: '🔬',
    placeholder: 'Ask a research question…',
    accent: 'from-rose-500 to-pink-600',
  },
]

export function getModeConfig(mode: string) {
  return CHAT_MODES.find((m) => m.id === mode) ?? CHAT_MODES[0]
}
