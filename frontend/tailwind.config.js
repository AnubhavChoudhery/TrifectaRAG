/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        chat: {
          bg: 'var(--chat-bg)',
          surface: 'var(--chat-surface)',
          sidebar: 'var(--chat-sidebar)',
          fg: 'var(--chat-fg)',
          'muted-fg': 'var(--chat-muted-fg)',
          border: 'var(--chat-border)',
          muted: 'var(--chat-muted)',
          accent: 'var(--chat-accent)',
          'accent-hover': 'var(--chat-accent-hover)',
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
