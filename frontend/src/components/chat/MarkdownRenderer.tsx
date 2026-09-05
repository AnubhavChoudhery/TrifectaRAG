import type { Components } from 'react-markdown'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import type { ImgHTMLAttributes } from 'react'
import { useState } from 'react'
import { normalizeMath } from '../../lib/normalizeMath'

type MarkdownRendererProps = {
  content: string
  className?: string
}

function MarkdownImage({ src, alt, ...props }: ImgHTMLAttributes<HTMLImageElement>) {
  const [failed, setFailed] = useState(false)

  if (failed) {
    return (
      <div className="my-4 rounded-xl border border-chat-border bg-chat-muted/60 px-4 py-3 text-sm text-chat-muted-fg">
        Image unavailable. Start the FastAPI backend with <code>python api.py</code> (port 8001), then refresh or ask again.
      </div>
    )
  }

  return (
    <img
      src={src ?? ''}
      alt={alt ?? 'PDF image'}
      className="my-4 max-h-[480px] w-auto max-w-full rounded-xl border border-chat-border bg-chat-muted object-contain"
      loading="lazy"
      onError={() => setFailed(true)}
      {...props}
    />
  )
}

const components: Components = {
  table({ children, ...props }) {
    return (
      <div className="my-4 overflow-x-auto rounded-lg border border-chat-border">
        <table className="min-w-full text-sm" {...props}>
          {children}
        </table>
      </div>
    )
  },
  thead({ children, ...props }) {
    return (
      <thead className="bg-chat-muted/60" {...props}>
        {children}
      </thead>
    )
  },
  th({ children, ...props }) {
    return (
      <th
        className="border-b border-chat-border px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-chat-muted-fg"
        {...props}
      >
        {children}
      </th>
    )
  },
  td({ children, ...props }) {
    return (
      <td className="border-b border-chat-border/60 px-4 py-2.5 text-chat-fg" {...props}>
        {children}
      </td>
    )
  },
  tr({ children, ...props }) {
    return (
      <tr className="transition hover:bg-chat-muted/30" {...props}>
        {children}
      </tr>
    )
  },
  pre({ children, ...props }) {
    return (
      <pre
        className="hljs my-4 overflow-x-auto rounded-xl border border-chat-border bg-[#0d1117] p-4 text-[13px] leading-relaxed text-slate-100"
        {...props}
      >
        {children}
      </pre>
    )
  },
  code({ className, children, ...props }) {
    const isBlock = className?.includes('language-')
    if (isBlock) {
      return (
        <code className={`${className ?? ''} font-mono`} {...props}>
          {children}
        </code>
      )
    }
    return (
      <code
        className="rounded-md bg-chat-muted px-1.5 py-0.5 font-mono text-[0.875em] text-chat-fg"
        {...props}
      >
        {children}
      </code>
    )
  },
  blockquote({ children, ...props }) {
    return (
      <blockquote
        className="my-4 border-l-4 border-chat-accent/60 bg-chat-accent/5 px-4 py-3 not-italic text-chat-fg/90"
        {...props}
      >
        {children}
      </blockquote>
    )
  },
  a({ children, href, ...props }) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="font-medium text-chat-accent underline decoration-chat-accent/40 underline-offset-2 hover:decoration-chat-accent"
        {...props}
      >
        {children}
      </a>
    )
  },
  img({ src, alt, ...props }) {
    return <MarkdownImage src={src} alt={alt} {...props} />
  },
  ul({ children, ...props }) {
    return (
      <ul className="my-3 list-disc space-y-1.5 pl-6" {...props}>
        {children}
      </ul>
    )
  },
  ol({ children, ...props }) {
    return (
      <ol className="my-3 list-decimal space-y-1.5 pl-6" {...props}>
        {children}
      </ol>
    )
  },
  h1({ children, ...props }) {
    return (
      <h1 className="mb-3 mt-5 text-xl font-bold text-chat-fg" {...props}>
        {children}
      </h1>
    )
  },
  h2({ children, ...props }) {
    return (
      <h2 className="mb-2 mt-5 text-lg font-semibold text-chat-fg" {...props}>
        {children}
      </h2>
    )
  },
  h3({ children, ...props }) {
    return (
      <h3 className="mb-2 mt-4 text-base font-semibold text-chat-fg" {...props}>
        {children}
      </h3>
    )
  },
  p({ children, ...props }) {
    return (
      <p className="my-2.5 leading-7 text-chat-fg/95" {...props}>
        {children}
      </p>
    )
  },
  hr({ ...props }) {
    return <hr className="my-6 border-chat-border" {...props} />
  },
}

export default function MarkdownRenderer({ content, className = '' }: MarkdownRendererProps) {
  return (
    <div className={`markdown-body prose-chat max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          [rehypeKatex, { throwOnError: false, strict: 'ignore' }],
          rehypeHighlight,
        ]}
        components={components}
      >
        {normalizeMath(content)}
      </ReactMarkdown>
    </div>
  )
}
