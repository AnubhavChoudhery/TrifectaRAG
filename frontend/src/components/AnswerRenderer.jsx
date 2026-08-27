import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { Check, Copy, FileText } from 'lucide-react'

export default function AnswerRenderer({ markdown }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(markdown).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    })
  }

  return (
    <section className="panel">
      <div className="section-heading">
        <div className="icon-tile">
          <FileText size={18} />
        </div>
        <div>
          <h2>Answer</h2>
          <p>Retrieved passages, formula text, and citations</p>
        </div>
        <button type="button" onClick={handleCopy} className="ghost-button ml-auto">
          {copied ? <Check size={15} /> : <Copy size={15} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>

      <div className="answer-prose">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: 'ignore' }]]}
          components={{
            table({ node, ...props }) {
              return (
                <div className="overflow-x-auto rounded-lg border border-slate-200">
                  <table {...props} />
                </div>
              )
            },
            th({ node, ...props }) {
              return <th className="border-b border-slate-200 bg-slate-100 px-3 py-2 text-left" {...props} />
            },
            td({ node, ...props }) {
              return <td className="border-b border-slate-100 px-3 py-2" {...props} />
            },
            pre({ node, ...props }) {
              return <pre className="overflow-x-auto rounded-lg bg-slate-950 p-4 text-sm text-slate-50" {...props} />
            },
            code({ node, inline, ...props }) {
              if (inline) {
                return <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-sm text-slate-800" {...props} />
              }
              return <code className="font-mono" {...props} />
            },
            blockquote({ node, ...props }) {
              return <blockquote className="border-l-4 border-teal-600 bg-teal-50 px-4 py-3 text-slate-800" {...props} />
            },
            h2({ node, ...props }) {
              return <h2 className="text-xl font-semibold text-slate-950" {...props} />
            },
            h3({ node, ...props }) {
              return <h3 className="mt-6 border-b border-slate-200 pb-2 text-base font-semibold text-slate-900" {...props} />
            },
          }}
        >
          {markdown}
        </ReactMarkdown>
      </div>
    </section>
  )
}
